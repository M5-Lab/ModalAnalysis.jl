export NMA


"""
NMA(eq::LammpsDump, ld::LammpsDump, pair_potential::Potential, potential_eng_MD, atom_masses, out_basepath, T_des, kB)
NMA(eq::LammpsDump, ld::LammpsDump, pair_potential::Potential, potential_eng_MD, atom_masses, out_basepath, T_des, kB, TEP_path)

Takes equilibrium positions and calcultes force constants and modal coupling constsants. These values
are to consturct the Taylor Effective Potential using data from a LAMMPS simulation. If `TEP_path` is passed
the freqs, phi & K3 will be loaded from another simulation to avoid re-calculating TEP parameters.

 - eq::LammpsDump : Equilibrium data parsed from dump file
 - ld::LammpsDump : Simulation data parsed from dump file
"""
function NMA(eq::LammpsDump, ld::LammpsDump, pair_potential::Potential, potential_eng_MD,
     atom_masses, out_basepath::String, gpu_device_id::Int)
   
    #Assumes 3D
    box_sizes = [ld.header_data["L_x"][2],ld.header_data["L_y"][2],ld.header_data["L_z"][2]]

    sys = SuperCellSystem(eq.data_storage, atom_masses, box_sizes, "x", "y", "z")

    # Initialize NMs to 3rd Order
    freqs_sq, phi, K3 = get_modal_data(sys, pair_potential; gpu_device_id = gpu_device_id)
    
    #Save mode data
    jldopen(joinpath(out_basepath, "TEP.jld2"), "w") do file
        file["freqs_sq"] = freqs_sq
        file["phi"] = phi
        file["K3"] = K3
    end

    NMA_loop(eq, ld, potential_eng_MD, atom_masses, out_basepath, freqs_sq, phi, K3)
end


function NMA(eq::LammpsDump, ld::LammpsDump, potential_eng_MD,
     atom_masses, out_basepath::String, TEP_path::String, gpu_device_id::Int)

    # Load data
    freqs_sq, phi, K3 = load(TEP_path, "freqs_sq", "phi", "K3")

    #Always save a copy of freqs and phi for post processing stuff
    jldopen(joinpath(out_basepath, "TEP.jld2"), "w") do file
        file["freqs_sq"] = freqs_sq
        file["phi"] = phi
    end
    
    NMA_loop(eq, ld, potential_eng_MD, atom_masses, out_basepath, freqs_sq, phi, K3)
end


function NMA_loop(eq::LammpsDump, ld::LammpsDump, 
     potential_eng_MD, atom_masses, out_basepath, freqs_sq, phi, K3)

    N_modes = length(freqs_sq)


    K3 = Float32.(K3)
    cuK3 = CUDA.CuArray(K3)
    cuQ = CUDA.zeros(N_modes)

    dump_file = open(ld.path, "r")
    initial_positions = Matrix(eq.data_storage[!, ["x","y","z"]])

    #Pre-allocate
    mode_potential_order3 = zeros(N_modes, ld.n_samples) #TODO: allocating might not work well for big data but H5 is slow at small writes
    total_eng_NM = zeros(ld.n_samples)

    @showprogress 10 "NMA Loop" for i in 1:ld.n_samples

        ld, dump_file = parse_next_timestep!(ld, dump_file) #this prevents CPU parallelization of this code data is stored in this object
        
        #Calculate displacements
        disp = Matrix(ld.data_storage[!, ["xu","yu","zu"]]) .- initial_positions
        for col in eachcol(disp) col .*= sqrt.(atom_masses) end
        disp_mw = reduce(vcat, eachrow(disp)) #TODO: should it be [xxxxxyyyyyzzz] or [xyzxyzxyz]??

        #Convert displacements to mode amplitudes.
        q_anharmonic = phi' * disp_mw;
        copyto!(cuQ, q_anharmonic)

        #Calculate energy from INMs at timestep i
        mode_potential_order3[:,i] .= 0.5.*(freqs_sq .* (q_anharmonic.^2)) .+ U_TEP3_n_CUDA(cuK3, cuQ)
        total_eng_NM[i] = @views sum(mode_potential_order3[:,i]) + potential_eng_MD[1]          

    end

    jldopen(joinpath(out_basepath, "ModeEnergies.jld2"), "w") do file
        file["mode_potential_order3"] = mode_potential_order3
        file["total_eng_NM"] = total_eng_NM
        file["potential_eng_MD"] = potential_eng_MD
    end

    close(dump_file)
end

