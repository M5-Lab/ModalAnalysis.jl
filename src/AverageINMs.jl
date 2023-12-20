export get_average_INMs

#cannot be parallelized easily with my existing infastructure
function get_average_INMs(inma::InstantaneousNormalModeAnalysis, calc::ForceConstantCalculator;
    verbose::Bool = true) where D

    N_modes = D*N_atoms
    avg_psi = zeros(N_modes, N_modes, N_modes)
    avg_dynmat = zeros(N_modes, N_modes)
    avg_forces = zeros(N_modes)
    
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
                avg_psi .+= third_order!(s, inma.potential, calc)
            end
            @async begin
                avg_forces .+= reduce(vcat, eachrow(Matrix(inma.ld.data_storage[!,["fx","fy","fz"]])))
                avg_dynmat .+= dynamical_matrix!(s, inma.potential, calc)
            end
        end


    end

    close(dump_file)
    
    avg_forces ./= inma.ld.n_samples
    avg_psi ./= inma.ld.n_samples
    avg_dynmat ./= inma.ld.n_samples
    

    return avg_forces, avg_dynmat, avg_psi

end
