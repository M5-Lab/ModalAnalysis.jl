export run

"""
    run(nma::NormalModeAnalysis)
    run(nma::NormalModeAnalysis, mcc_block_size::Integer)
    run(nma::NormalModeAnalysis, TEP_path::String)
"""

# Calculate MCC fresh
function run(nma::NormalModeAnalysis)

    # Initialize NMs to 3rd Order
    freqs_sq, phi, dynmat, K3 = get_modal_data(nma)
    
    #Save mode data
    jldopen(joinpath(nma.simulation_folder, "TEP.jld2"), "w"; compress = true) do file
        file["freqs_sq"] = freqs_sq
        file["phi"] = phi
        file["K3"] = K3
        file["dynmat"] = dynmat.values
    end

    NMA_loop(nma, nma.simulation_folder, freqs_sq, phi, K3)
end

#Calculate MCC with blocked-approach to save RAM
function run(nma::NormalModeAnalysis, mcc_block_size::Integer)

    # Initialize NMs to 3rd Order
    freqs_sq, phi, dynmat, K3 = get_modal_data(nma, mcc_block_size)
    
    #Save mode data
    jldopen(joinpath(nma.simulation_folder, "TEP.jld2"), "w"; compress = true) do file
        file["freqs_sq"] = freqs_sq
        file["phi"] = phi
        file["K3"] = K3
        file["dynmat"] = dynmat.values
    end

    NMA_loop(nma, nma.simulation_folder, freqs_sq, phi, K3)
end

#Re-use MCC from a previous simulation
function run(nma::NormalModeAnalysis, TEP_path::String)
    
    f = jldopen(TEP_path, "r"; parallel_read = true)
    freqs_sq = f["freqs_sq"]
    phi = f["phi"]
    K3 = f["K3"]
    dynmat = f["dynmat"]
    close(f)

    #Always save a copy of freqs and phi for post processing stuff
    jldopen(joinpath(nma.simulation_folder, "TEP.jld2"), "w") do file
        file["freqs_sq"] = freqs_sq
        file["phi"] = phi
        file["dynmat"] = dynmat
    end

    timer = TimerOutput()
    @timeit timer "NMA_loop" NMA_loop(nma, nma.simulation_folder, freqs_sq, phi, K3)

    open(joinpath(nma.simulation_folder, "timings.txt"), "a+") do f
        print_timer(f, timer)
    end
end 

function NMA_loop(nma::NormalModeAnalysis, out_path::String, freqs_sq, phi, K3)

    N_modes = length(freqs_sq)
    N_atoms = length(nma.atom_masses)

    mass_sqrt = sqrt.(nma.atom_masses)

    K3 = Float32.(K3) #TODO MAKE TYPE A PARAMETER FOR MCC3 TO SAVE TIME CASTING (matters more for INMS)
    cuK3 = CUDA.CuArray(K3)
    
    #Pre-allocate intermediate data_storage
    disp = zeros(N_atoms, 3); disp_mw = zeros(N_modes)
    q = zeros(Float32,N_modes)
    cuQ = CUDA.zeros(N_modes)

    dump_file = open(nma.ld.path, "r")
    posn_cols = [nma.ld.col_idxs["xu"],nma.ld.col_idxs["yu"],nma.ld.col_idxs["zu"]]
    initial_positions = Matrix(nma.eq.data_storage[!, ["xu","yu","zu"]])
    current_positions = zeros(size(initial_positions))
    
    #Pre-allocate output arrays
    mode_potential_order3 = zeros(N_modes, nma.ld.n_samples)
    total_eng_NM = zeros(nma.ld.n_samples)

    for i in 1:nma.ld.n_samples

        parse_next_timestep!(current_positions, nma.ld, dump_file, posn_cols)
        
        #Calculate displacements
        disp .= current_positions .- initial_positions

        disp .*= mass_sqrt 
        disp_mw .= reduce(vcat, eachrow(disp))

        #Convert displacements to mode amplitudes.
        mul!(q, phi', disp_mw)
        copyto!(cuQ, q)

        #Calculate energy from INMs at timestep i
        mode_potential_order3[:,i] .= 0.5.*(freqs_sq .* (q.^2)) .+ Array(U_TEP3_n_CUDA(cuK3, cuQ)) #&slowest step, can I make TensorOpt faster? just do on CPU
        total_eng_NM[i] = @views sum(mode_potential_order3[:,i]) + nma.eq_pot_eng
    end 

    timer2 = TimerOutput()

    @timeit timer2 "Energy Write" jldopen(joinpath(out_path, "ModeEnergies.jld2"), "w"; compress = true) do file
        file["mode_potential_order3"] = mode_potential_order3
        file["total_eng_NM"] = total_eng_NM
        file["pot_eng_MD"] = nma.pot_eng_MD
    end

    open(joinpath(nma.simulation_folder, "timings.txt"), "a+") do f
        print_timer(f, timer2)
    end

    close(dump_file)
end


