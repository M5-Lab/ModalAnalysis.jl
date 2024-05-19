export get_average_INMs

#cannot be parallelized easily with my existing infastructure
#ASSUMES 3D SYSTEM, and everything passed is float32
function get_average_INMs(inma::InstantaneousNormalModeAnalysis, calc::ForceConstantCalculator;
    verbose::Bool = true, ncheckpoints::Int = 1, filename = "AvgINM.jld2", T = Float32)

    N_atoms = n_atoms(get_sys(inma))
    N_modes = 3*N_atoms
    avg_psi = zeros(T, N_modes, N_modes, N_modes)
    psi_storage = similar(avg_psi)
    avg_dynmat = zeros(T, N_modes, N_modes)
    dynmat_storage = similar(avg_dynmat)
    avg_forces = zeros(T, N_modes)
    tmp_forces = zeros(T, N_modes)
    
    dump_file = open(inma.ld.path, "r")

    #Calculate indicies to save data to disk
    checkpoints = collect((inma.ld.n_samples ÷ ncheckpoints) .* (1:(ncheckpoints-1)))
    checkpoint_counter = 1

    verbose && @info "Starting temp: $(inma.temperature) with $(inma.ld.n_samples) samples and $(ncheckpoints) checkpoints."
    for i in 1:inma.ld.n_samples
        verbose && begin i % 50 == 0 && @info "Sample $i, T: $(inma.temperature)" end
        #Parse data from dump file into inma.ld.data_storage
        parse_next_timestep!(inma.ld, dump_file, float_type = T)
    
        #Update positions in inma
        box_sizes = [inma.ld.header_data["L_x"][2], inma.ld.header_data["L_y"][2], inma.ld.header_data["L_z"][2]]
        s = SuperCellSystem(inma.ld.data_storage, inma.atom_masses, box_sizes, "x", "y", "z")
        
        avg_psi .+= third_order!(psi_storage, s, inma.potential, calc)

        avg_forces .+= reduce(vcat, eachrow(Matrix(inma.ld.data_storage[!,["fx","fy","fz"]])))
        avg_dynmat .+= dynamical_matrix!(dynmat_storage, s, inma.potential, calc)

        if i ∈ checkpoints
            verbose && @info "Saving Checkpoint $i"
            @assert i != 0 "Bug in checkpointing code. Checkpointing at 0th sample."

            #Re-use storage to calculate checkpoint-data
            tmp_forces .= avg_forces ./ i
            dynmat_storage .= avg_dynmat ./ i
            psi_storage .= avg_psi ./ i
            # freqs_sq, phi = get_modes(dynmat_storage)

            checkpoint_dir = joinpath(inma.simulation_folder, "AvgINM_Checkpoint$(checkpoint_counter)")
            mkpath(checkpoint_dir)

            jldsave(joinpath(checkpoint_dir, filename), 
                f0 = tmp_forces,
                dynmat = dynmat_storage, 
                F3 = psi_storage,
                nsamples = i
            )

            checkpoint_counter += 1
        end

        fill!(psi_storage, T(0.0))
        fill!(dynmat_storage, T(0.0))
    end

    close(dump_file)
    
    avg_forces ./= inma.ld.n_samples
    avg_psi ./= inma.ld.n_samples
    avg_dynmat ./= inma.ld.n_samples
    

    return avg_forces, avg_dynmat, avg_psi

end
