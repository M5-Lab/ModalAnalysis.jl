export GenerateConfigs

function energy_from_potential(temperatures, all_configs, sys_eq::SuperCellSystem, pot::Potential, energy_unit)
    n_configs = size(all_configs,3)
    N_atoms = n_atoms(sys_eq)

    U = zeros(Float32, length(temperatures), n_configs) * energy_unit
    
    for i in eachindex(temperatures)
        @info "Calculating Energies for $(temperatures[i]) K"
        for (j,config) in enumerate(eachcol(all_configs[i, :, :]))
            posns = [SVector{3}(config[3*(i-1) + 1], config[3*(i-1) + 2], config[3*(i-1) + 3]) for i in 1:N_atoms]
            U[i,j] = energy_loop(pot, posns, sys_eq.box_sizes_SC, N_atoms, pot.r_cut)
        end
    end
    return U
end

function energy_from_tep2(temperatures, all_configs, eq_positions, ifc2::Matrix, energy_unit)
    n_configs = size(all_configs,3)

    U = zeros(Float32, length(temperatures), n_configs) * energy_unit
    for i in eachindex(temperatures)
        @info "Calculating TEP Energies for $(temperatures[i]) K"
        for (j,config) in enumerate(eachcol(all_configs[i, :, :]))
            disp = config .- eq_positions
            U[i,j] = (transpose(disp) * ifc2) * disp
        end
    end
    return 0.5 .* U
end


function GenerateConfigs(sys_eq::SuperCellSystem{3}, temperatures::AbstractVector,
    freqs::AbstractVector, phi::AbstractMatrix, n_configs::Int, energy_unit::Unitful.FreeUnits;
    mode = :quantum, nthreads = Threads.nthreads(), pot::Union{Potential, Nothing} = nothing,
    ifc2::Union{Matrix, Nothing} = nothing)

    if mode ∉ [:quantum, :classical]
        error("mode must be :quantum or :classical")
    end

    if energy_unit == u"eV"  
        kB = uconvert(u"eV/K", Unitful.k)
        hbar = uconvert(u"eV * s", Unitful.ħ)
    elseif energy_unit == u"kcal/mol"
        kB = uconvert(u"kcal * mol^-1 * K^-1", Unitful.k*Unitful.Na)
        hbar = uconvert(u"kcal * mol^-1 * s", Unitful.ħ*Unitful.Na)
    else
        error("Unknown unit system")
    end

    if !isnothing(pot)
        @assert length_unit(pot) == unit(sys_eq.atoms.position[1][1])
    end
    
    eq_positions = reduce(vcat, positions(sys_eq))
    atom_masses = masses(sys_eq)

    N_dof = length(eq_positions)
    N_atoms = n_atoms(sys_eq)

    all_configs = zeros(Float32, length(temperatures), N_dof, n_configs) * length_unit(pot)
    z = randn(Float32, length(temperatures), N_dof, n_configs)

    rtm_idxs = rigid_translation_modes(freqs, 3)

    for (i,temp) in enumerate(temperatures)
        @info "Generating Configurations for $temp K"
        @views generate_configs!(all_configs[i, :, :], N_atoms, freqs, phi, z[i,:,:], eq_positions,
                                 rtm_idxs, atom_masses, temp, kB, hbar, mode; nthreads = nthreads)
    end

    if isnothing(pot) && isnothing(ifc2)
        return all_configs
    elseif isnothing(pot) && !isnothing(ifc2)
        # Calculate energy from second-order TEP
        U_TEP = energy_from_tep2(temperatures, all_configs, eq_positions, ifc2, energy_unit)
        return all_configs, U_TEP
    elseif !isnothing(pot) && isnothing(ifc2)
        U = energy_from_potential(temperatures, all_configs, sys_eq, pot, energy_unit)
        return all_configs, U
    else
        U_TEP = energy_from_tep2(temperatures, all_configs, eq_positions, ifc2, energy_unit)
        U = energy_from_potential(temperatures, all_configs, sys_eq, pot, energy_unit)
        return all_configs, U, U_TEP
    end

end