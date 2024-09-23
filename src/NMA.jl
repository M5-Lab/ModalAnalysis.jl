export run

"""
    run(nma::NormalModeAnalysis; mcc_block_size::Integer = nothing, order::Int = 3)
    run(nma::NormalModeAnalysis, TEP_path::String; energy_block_size::Integer = nothing, order::Int = 3)
"""

# Calculate MCC fresh
function run(nma::NormalModeAnalysis; mcc_block_size::Union{Integer, Nothing} = nothing, order::Int = 3)

    @assert nma.calc !== nothing "ForceConstantCalculator not passed to nma class"

    # Initialize NMs to 3rd Order
    if mcc_block_size === nothing
        freqs_sq, phi, dynmat, K3 = get_modal_data(nma)
    else
        freqs_sq, phi, dynmat, K3 = get_modal_data(nma, mcc_block_size)
    end
    
    #Save mode data
    jldopen(joinpath(nma.simulation_folder, "TEP.jld2"), "w"; compress = true) do file
        file["freqs_sq"] = freqs_sq
        file["phi"] = phi
        file["K3"] = K3
        file["dynmat"] = dynmat.values
    end

    NMA_loop(nma, nma.simulation_folder, freqs_sq, phi, K3, U_TEP3_n_CUDA, order)

    return nothing
end


#Re-use MCC from a previous simulation
# could be nma::Union{NormalModeAnalysis, MonteCarloNormalModeAnalysis
function run(nma::NormalModeAnalysis,
     TEP_path::String; energy_block_size::Union{Integer, Nothing} = nothing, order::Int = 3)
    
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

    if energy_block_size === nothing
        NMA_loop(nma, nma.simulation_folder, freqs_sq, phi, K3, U_TEP3_n_CUDA, order)
    else
        U_TEP3_function = (cuK3, cuQ) -> U_TEP3_n_CUDA(cuK3, cuQ, energy_block_size)
        NMA_loop(nma, nma.simulation_folder, freqs_sq, phi, K3, U_TEP3_function, order)
    end


    return nothing
end 


function NMA_loop(nma::NormalModeAnalysis, out_path::String, freqs_sq, phi, K3, U_TEP3_func::Function, order::Int)

    if order < 2
        error("Order must be 2 or 3")
    elseif order >=4 
        error("Order must be 2 or 3")
    end

    N_modes = length(freqs_sq)
    N_atoms = length(nma.atom_masses)

    mass_sqrt = sqrt.(nma.atom_masses)

    K3 = Float32.(K3)
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
    mode_potential_energy = zeros(N_modes, nma.ld.n_samples)
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
        mode_potential_energy[:,i] .= 0.5.*(freqs_sq .* (q.^2))
        if order == 3
            mode_potential_energy[:,i] .+= Array(U_TEP3_func(cuK3, cuQ))
        end
        # mode_potential_energy[:,i] .= 0.5.*(freqs_sq .* (q.^2)) .+ Array(U_TEP3_func(cuK3, cuQ))
        total_eng_NM[i] = @views sum(mode_potential_energy[:,i]) + nma.eq_pot_eng
    end 

    jldopen(joinpath(out_path, "ModeEnergies.jld2"), "w"; compress = true) do file
        file["mode_potential_energy"] = mode_potential_energy
        file["total_eng_NM"] = total_eng_NM
        file["pot_eng_MD"] = nma.pot_eng_MD
        file["order"] = order
    end


    close(dump_file)
    return nothing
end


# function NMA_loop(nma::MonteCarloNormalModeAnalysis, out_path::String, freqs_sq, phi, K3, U_TEP3_func::Function, order::Int)
    
#     if order < 2
#         error("Order must be 2 or 3")
#     elseif order >=4 
#         error("Order must be 2 or 3")
#     end

#     N_modes = length(freqs_sq)
#     N_atoms = length(nma.atom_masses)

#     mass_sqrt = sqrt.(nma.atom_masses)

#     K3 = Float32.(K3)
#     cuK3 = CUDA.CuArray(K3)
    
#     #Pre-allocate intermediate data_storage
#     disp = zeros(N_atoms, 3); disp_mw = zeros(N_modes)
#     q = zeros(Float32,N_modes)
#     cuQ = CUDA.zeros(N_modes)

#     initial_positions = Matrix(nma.sim.reference_positions)
#     current_positions = zeros(size(initial_positions))
    
#     #Pre-allocate output arrays
#     mode_potential_energy = zeros(N_modes, nma.sim.n_steps)
#     total_eng_NM = zeros(sim.n_steps)

#     for i in 1:nma.sim.n_steps

#         parse_next_timestep!(current_positions, nma, dump_file, posn_cols)
        
#         #Calculate displacements
#         disp .= current_positions .- initial_positions

#         disp .*= mass_sqrt 
#         disp_mw .= reduce(vcat, eachrow(disp))

#         #Convert displacements to mode amplitudes.
#         mul!(q, phi', disp_mw)
#         copyto!(cuQ, q)

#         #Calculate energy from INMs at timestep i
#         mode_potential_energy[:,i] .= 0.5.*(freqs_sq .* (q.^2))
#         if order == 3
#             mode_potential_energy[:,i] .+= Array(U_TEP3_func(cuK3, cuQ))
#         end
#         # mode_potential_energy[:,i] .= 0.5.*(freqs_sq .* (q.^2)) .+ Array(U_TEP3_func(cuK3, cuQ))
#         total_eng_NM[i] = @views sum(mode_potential_energy[:,i])
#     end


#     jldopen(joinpath(out_path, "ModeEnergies.jld2"), "w"; compress = true) do file
#         file["mode_potential_energy"] = mode_potential_energy
#         file["total_eng_NM"] = total_eng_NM
#         file["pot_eng_MD"] = nma.pot_eng_MD
#         file["order"] = order
#     end

#     return nothing
# end

