

function bose_einstein(freq, temp, kB, hbar)
    return 1 / (exp((hbar * freq) / (kB * temp)) - 1)
end


function quantum_amplitude(freq, mass, temp, hbar)
    n = bose_einstein(freq, temp, kB, hbar)
    numerator = hbar * (2*n + 1)
    denominator = 2 * mass * freq
    return sqrt(numerator / denominator)
end

function classical_amplitude(freq, mass, kB, temp)
    return sqrt((kB*temp)/mass) / freq
end

function self_consistent_IFC_loop(sys_eq::SuperCellSystem, calc::ForceConstantCalculator,
                                  temp, pot::Potential, n_configs::Int, mode::Symbol) where D

    N_atoms = n_atoms(sys_eq)
    N_dof = D * N_atoms

    # This should never be much more than ~50 MB in practice
    z = randn(Float32, N_dof, n_configs) #* do I need to re-sample this for each atom? 

    # Calculate 0K IFCs to initialize loop
    dynmat = dynamical_matrix(sys_eq, pot, calc)
    T = eltype(dynmat)
    freqs_sq, phi = get_modes(dynmat, D)
    freqs = sqrt.(Complex.(freqs_sq))

    atom_masses = masses(sys_eq)
    box_sizes = copy(box_sizes(sys_eq))

    # Figure out proper units for constants
    if energy_unit(pot) == u"eV"
        kB = ustrip(u"eV/K", Unitful.k)
        hbar = ustrip(u"eV * s", Unitful.ħ)
    elseif energy_unit(pot) == u"kcal/mol"
        kB = ustrip(u"kcal * mol^-1 * K^-1", Unitful.k*Unitful.Na)
        hbar = ustrip(u"kcal * mol^-1 * s", Unitful.ħ*Unitful.Na)
    else
        error("Unknown unit system")
    end

    # all_configs = zeros(Float32, N_dof, n_configs)
    current_config = zeros(Float32, N_atoms, 3)
    U = zeros(n_configs)
    
    for n in 1:n_configs

        # Generate configurations with current set of IFCs
        for i in 1:N_atoms
            for α in 1:D
                for m in 1:N_modes
                    if mode == :quantum
                        A = quantum_amplitude(freqs[m], atom_masses[i], temp, hbar)
                    elseif mode == :classical
                        A = classical_amplitude(freqs[m], atom_masses[i], kB, temp)
                    else
                        error("Unknown mode")
                    end

                    # @views configs[ii, n] = A * z[m, n] * phi[ii, m]
                    @views current_config[i, α] = A * z[m, n] * phi[ii, m]
                end
            end
        end

        # Update IFCs
        fill!(dynmat, T(0.0))
        sys = SuperCellSystem(current_config, atom_maxxes, box_sizes)
        dynmat = dynamical_matrix!(dynmat, sys, pot, calc)
        fres_sq, phi = get_modes(dynmat, D) #*allocates
        freqs .= sqrt.(Complex.(fres_sq))

        # Calculate convergence metric
        #& does z[m,n] * phi[ii,m] = q_n
        # U[n] = 0.5 * (trans)
    end
    
    return configs
end