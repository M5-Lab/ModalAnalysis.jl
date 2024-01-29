export get_average_INMs

#cannot be parallelized easily with my existing infastructure
#ASSUMES 3D SYSTEM
function get_average_INMs(inma::InstantaneousNormalModeAnalysis, calc::ForceConstantCalculator;
    verbose::Bool = true)

    N_atoms = n_atoms(get_sys(inma))
    N_modes = 3*N_atoms
    avg_psi = zeros(Float32, N_modes, N_modes, N_modes)
    psi_storage = similar(avg_psi)
    avg_dynmat = zeros(Float32, N_modes, N_modes)
    dynmat_storage = similar(avg_dynmat)
    avg_forces = zeros(Float32, N_modes)
    
    dump_file = open(inma.ld.path, "r")

    verbose && @info "$(inma.ld.n_samples) samples"
    for i in 1:inma.ld.n_samples
        verbose && @info "$i"
        #Parse data from dump file into inma.ld.data_storage
        parse_next_timestep!(inma.ld, dump_file)
    
        #Update positions in inma
        s = get_sys(inma)
        box_sizes = [inma.ld.header_data["L_x"][2], inma.ld.header_data["L_y"][2], inma.ld.header_data["L_z"][2]]
        s = SuperCellSystem(inma.ld.data_storage, inma.atom_masses, box_sizes, "x", "y", "z")
        
        @sync begin
            @async begin
                avg_psi .+= third_order!(psi_storage, s, inma.potential, calc)
                fill!(psi_storage, 0.0f0)
            end
            @async begin
                avg_forces .+= reduce(vcat, eachrow(Matrix(inma.ld.data_storage[!,["fx","fy","fz"]])))
                avg_dynmat .+= dynamical_matrix!(dynmat_storage,s, inma.potential, calc)
                fill!(dynmat_storage, 0.0f0)
            end
        end


    end

    close(dump_file)
    
    avg_forces ./= inma.ld.n_samples
    avg_psi ./= inma.ld.n_samples
    avg_dynmat ./= inma.ld.n_samples
    

    return avg_forces, avg_dynmat, avg_psi

end
