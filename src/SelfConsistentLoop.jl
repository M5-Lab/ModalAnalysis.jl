function bose_einstein(freq, temp, kB, hbar)
    return 1 / (exp((hbar * freq) / (kB * temp)) - 1)
end


function quantum_amplitude(freq, mass, temp, kB, hbar)
    nᵢ = bose_einstein(freq, temp, kB, hbar)
    return sqrt((hbar * (2*nᵢ + 1)) / (2 * mass * freq))
end

function classical_amplitude(freq, mass, kB, temp)
    return sqrt((kB*temp)/mass) / freq
end

# Checks for imaginary modes and 0 frequency modes
function flag_modes!(flags::BitVector, freqs, tol = 1e-7)
    fill!(flags, false)

    for (i, freq) in enumerate(freqs)
        if abs(imag(freq)) > tol
            flags[i] = true
        elseif abs(real(freq)) < tol
            flags[i] = true
        end
    end

    return flags
end

struct ImaginaryModeError <: Exception
    msg::String
end

function imaginary_mode_present(freqs::AbstractVector, tol = 1e-6)
    for freq in freqs
        if abs(imag(freq)) > tol
            return true
        end
    end
    return false
end

# Assumes there are D modes with freuqency 0
function rigid_translation_modes(freqs, D, tol = 1e-7)
    idx_rt = sortperm(abs.(freqs))
    return idx_rt[1:D]
end

struct SelfConsistentConfigs
    configs::Matrix{Float32}
    freq_checkpoints::Matrix{Float32}
    kB::Float32
    hbar::Float32
    temp::Float32
    n_iters::Int
end

n_configs(sc::SelfConsistentConfigs) = size(sc.configs, 2)


function self_consistent_IFC_loop(sys_eq::SuperCellSystem{D}, calc::ForceConstantCalculator,
                                  temp, pot::Potential, n_configs::Int, n_iters::Int,
                                  mode::Symbol; save_every::Int = 25,
                                  nthreads::Int = Threads.nthreads())::Result{SelfConsistentConfigs, ImaginaryModeError} where D

    @assert length_unit(pot) == unit(positions(sys_eq)[1][1])

    N_atoms = n_atoms(sys_eq)
    N_dof = D * N_atoms

    # This should never be much more than ~50 MB in practice
    T = Float32
    z = randn(T, N_dof, n_configs) #* should this be different per atom or per mode?

    # Calculate 0K IFCs to initialize loop
    dynmat = zeros(T, N_dof, N_dof)
    dynamical_matrix!(dynmat, sys_eq, pot, calc)
    freqs_sq, phi = get_modes(dynmat, D)
    freqs = sqrt.(Complex.(freqs_sq)) #./ (2π) #* need to divide by 2pi?

    # Check for imaginary and 0 frequency modes
    if imaginary_mode_present(freqs)
        return ImaginaryModeError("Imaginary mode found in initial modes (Zero Kelvin IFCs)")
    end

    rtm_idxs = rigid_translation_modes(freqs, D)
    freqs = real(freqs)

    atom_masses = ustrip.(masses(sys_eq))
    eq_positions = ustrip.(reduce(vcat, positions(sys_eq)))
    box_sizes = copy(sys_eq.box_sizes_SC)
    L_unit = length_unit(pot)

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
    n_checkpoints = cld(n_iters, save_every)
    freq_checkpoints = zeros(Float32, N_dof, n_checkpoints + 1)
    avg_dynmat = zeros(T, N_dof, N_dof)

    # Save initial data into checkpoints
    freq_checkpoints[:, 1] .= freqs
    checkpoint_idx = 2
    
    for iter in 1:n_iters
        bar = ProgressBar(1:n_configs; printing_delay = 0.1)
        set_description(bar, "Making Configs, Iter: $iter")
        for n in bar
            # Generate configurations with current set of IFCs
            @tasks for i in 1:N_atoms 
                @set ntasks = nthreads
                for α in 1:D #* could just vectorize over this, since amplitudes are per atom
                    ii = D*(i-1) + α #atom idx
                    for m in 1:N_dof #mode idx

                        #* IGNORE RIGID TRANSLATION MODES?
                        if m in rtm_idxs
                            continue
                        end

                        if mode == :quantum
                            A = quantum_amplitude(freqs[m], atom_masses[i], temp, kB, hbar)
                        elseif mode == :classical
                            A = classical_amplitude(freqs[m], atom_masses[i], kB, temp)
                        else
                            error("Unknown mode")
                        end

                        @views configs[ii, n] += (A * z[m, n] * phi[ii, m])
                    end
                end
            end
            # Just calculated displacements before, add eq positions to get configuration
            configs[:, n] .+= eq_positions
        end
        
         # Calculate tempearture dependent IFCs from generated configurations
        bar = ProgressBar(eachcol(configs), printing_delay = 0.2)
        set_description(bar, "Calculating AvgIFCs, Iter: $iter")
        for config in bar
            fill!(dynmat, T(0.0))
            xs = [config[D*(i-1) + 1 : D*i] * L_unit for i in 1:N_atoms]  #*allocates
            sys = SuperCellSystem(xs, atom_masses, box_sizes)
            avg_dynmat .+= dynamical_matrix!(dynmat, sys, pot, calc)
        end

        avg_dynmat ./= n_configs
        fres_sq, phi = get_modes(avg_dynmat, D) #*allocates
        freqs .= sqrt.(Complex.(fres_sq)) #./ (2π)
        rtm_idxs .= rigid_translation_modes(freqs, D)


        if imaginary_mode_present(freqs)
            return ImaginaryModeError("Imaginary mode present on iteration $(iter)")
        end

        if iter % save_every == 0
            freq_checkpoints[:, checkpoint_idx] .= freqs
        end

        # Reset random numbers
        randn!(z)
    end

    return SelfConsistentConfigs(configs, freq_checkpoints, kB, hbar, temp, n_iters)
end