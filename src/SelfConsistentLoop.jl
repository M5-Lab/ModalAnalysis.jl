using Random


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

function self_consistent_IFC_loop(sys_eq::SuperCellSystem{D}, calc::ForceConsatntCalculator,
                                  temp::Real, pot::Potential, n_configs::Int, 
                                  outpath::String, mode::Symbol) where D

    N_atoms = n_atoms(sys_eq)
    N_dof = D * N_atoms

    # This should never be much more than ~50 MB in practice
    z = randn(Float32, N_dof, n_configs) #* do I need to re-sample this for each atom? 

    # Calculate 0K IFCs to initialize loop
    dynmat = dynamical_matrix(sys_eq, pot, calc)
    fres_sq, phi = get_modes(dynmat, D)
    freqs = sqrt.(Complex.(fres_sq))

    # Create version of masses that is D * N_atoms long
    atom_masses = masses(sys_eq)

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

    configs = zeros(Float32, N_dof, n_configs)
    
    for n in 1:n_configs

        for i in 1:N_atoms
            for α in 1:D
                ii = D*(i-1) + α

                for m in 1:N_modes
                    if mode == :quantum
                        A = quantum_amplitude(freqs[m], atom_masses[i], temp, hbar)
                    elseif mode == :classical
                        A = classical_amplitude(freqs[m], atom_masses[i], kB, temp)
                    else
                        error("Unknown mode")
                    end

                    @views configs[ii, n] = A * z[m, n] * phi[ii, m]
                end
            end
        end


    end
    
    return configs
end