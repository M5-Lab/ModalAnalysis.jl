export get_average_INMs

#cannot be parallelized easily with my existing infastructure
#ASSUMES 3D SYSTEM
function get_average_INMs(inma::InstantaneousNormalModeAnalysis, calc::ForceConstantCalculator;
    verbose::Bool = true, ncheckpoints::Int = 1, filename = "AvgINM.jld2")

    N_atoms = n_atoms(get_sys(inma))
    N_modes = 3*N_atoms
    avg_psi = zeros(Float32, N_modes, N_modes, N_modes)
    psi_storage = similar(avg_psi)
    avg_dynmat = zeros(Float32, N_modes, N_modes)
    dynmat_storage = similar(avg_dynmat)
    avg_forces = zeros(Float32, N_modes)
    tmp_forces = zeros(Float32, N_modes)
    
    dump_file = open(inma.ld.path, "r")

    #Calculate indicies to save data to disk
    checkpoints = (inma.ld.n_samples ÷ ncheckpoints) .* (1:(ncheckpoints-1))
    next_checkpoint_idx = 1


    verbose && @info "Starting temp: $(inma.temperature) with $(inma.ld.n_samples) samples and $(ncheckpoints) checkpoints."
    for i in 1:inma.ld.n_samples
        verbose && begin i % 100 == 0 && @info "Sample $i, T: $(inma.temperature)" end
        #Parse data from dump file into inma.ld.data_storage
        parse_next_timestep!(inma.ld, dump_file)
    
        #Update positions in inma
        s = get_sys(inma)
        box_sizes = [inma.ld.header_data["L_x"][2], inma.ld.header_data["L_y"][2], inma.ld.header_data["L_z"][2]]
        s = SuperCellSystem(inma.ld.data_storage, inma.atom_masses, box_sizes, "x", "y", "z")
        
        avg_psi .+= third_order!(psi_storage, s, inma.potential, calc)

        avg_forces .+= reduce(vcat, eachrow(Matrix(inma.ld.data_storage[!,["fx","fy","fz"]])))
        avg_dynmat .+= dynamical_matrix!(dynmat_storage,s, inma.potential, calc)

        if i == checkpoints[next_checkpoint_idx]
            next_checkpoint_idx += 1
            verbose && @info "Saving Checkpoint $i"

            #Re-use storage to calculate checkpoint-data
            tmp_forces .= avg_forces ./ inma.ld.n_samples
            dynmat_storage .= avg_dynmat ./ inma.ld.n_samples
            psi_storage .= avg_psi ./ inma.ld.n_samples
            freqs_sq, phi = get_modes(dynmat_storage)

            checkpoint_dir = joinpath(out_path, "AvgINM_Checkpoint$(next_checkpoint_idx)")
            mkpath(checkpoint_dir)

            jldsave(joinpath(checkpoint_dir, filename), 
                f0 = tmp_forces,
                dynmat = dynmat_storage, 
                F3 = psi_storage,
                freqs_sq = freqs_sq, phi = phi,
                nsamples = i
            )
        end

        fill!(psi_storage, 0.0f0)
        fill!(dynmat_storage, 0.0f0)

    end

    close(dump_file)
    
    avg_forces ./= inma.ld.n_samples
    avg_psi ./= inma.ld.n_samples
    avg_dynmat ./= inma.ld.n_samples
    

    return avg_forces, avg_dynmat, avg_psi

end
