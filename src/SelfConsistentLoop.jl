
function bose_einstein(freq, temp, kB, hbar)
    x =  upreferred((hbar * freq) / (kB * temp))
    return 1 / (exp(x) - 1)
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
        if ustrip(abs(imag(freq))) > tol
            return true
        end
    end
    return false
end

# Assumes there are D modes with freuqency 0
function rigid_translation_modes(freqs, D)
    idx_rt = sortperm(abs.(freqs))
    return SVector(idx_rt[1:D]...)
end

struct SelfConsistentConfigs{C,K,H,T}
    configs::Matrix{C}
    freq_checkpoints::Matrix{Float32}
    dynmat_checkpoints::Array{Float32, 3}
    checkpoint_idxs::Vector{Int}
    kB::K
    hbar::H
    temp::T
    n_iters::Int
end

n_configs(sc::SelfConsistentConfigs) = size(sc.configs, 2)

function generate_configs!(configs::AbstractMatrix, N_atoms::Int, freqs::AbstractVector,
                         phi::AbstractMatrix, z::AbstractMatrix, eq_positions::AbstractVector,
                         rtm_idxs, atom_masses, temp, kB, hbar, mode::Symbol;
                         nthreads::Int = Threads.nthreads())

    N_dof, n_configs = size(configs)
    D = length(rtm_idxs)
    L_unit = unit(configs[firstindex(configs)])

    # bar = ProgressBar(1:n_configs; printing_delay = 0.1)
    # set_description(bar, "Making Configs")
    for n in 1:n_configs
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
                        A = uconvert(L_unit, A)
                    elseif mode == :classical
                        A = classical_amplitude(freqs[m], atom_masses[i], kB, temp)
                        A = uconvert(L_unit, A) # shouldnt matter here
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

    return configs
end

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

    freq_unit = sqrt(energy_unit(pot) / (length_unit(pot)^2) / unit(first(masses(sys_eq))))
    freqs = sqrt.(Complex.(freqs_sq)) * freq_unit

    # Check for imaginary and 0 frequency modes
    if imaginary_mode_present(freqs)
        return ImaginaryModeError("Imaginary mode found in initial modes (Zero Kelvin IFCs)")
    end

    rtm_idxs = rigid_translation_modes(freqs, D)
    freqs = real(freqs)

    atom_masses = masses(sys_eq)
    eq_positions = reduce(vcat, positions(sys_eq))
    box_sizes = copy(sys_eq.box_sizes_SC)
    L_unit = length_unit(pot)

    # Figure out proper units for constants
    if energy_unit(pot) == u"eV"
        kB = uconvert(u"eV/K", Unitful.k)
        hbar = uconvert(u"eV * s", Unitful.ħ)
    elseif energy_unit(pot) == u"kcal/mol"
        kB = uconvert(u"kcal * mol^-1 * K^-1", Unitful.k*Unitful.Na)
        hbar = uconvert(u"kcal * mol^-1 * s", Unitful.ħ*Unitful.Na)
    else
        error("Unknown unit system")
    end

    configs = zeros(Float32, N_dof, n_configs) * L_unit
    n_checkpoints = n_iters ÷ save_every
    freq_checkpoints = zeros(Float32, N_dof, n_checkpoints + 2)
    dynmat_checkpoints = zeros(Float32, N_dof, N_dof, n_checkpoints + 2)
    checkpoint_idxs = [1]
    avg_dynmat = zeros(T, N_dof, N_dof)

    # Save initial data into checkpoints
    freq_checkpoints[:, 1] .= ustrip.(freqs)
    dynmat_checkpoints[:, :, 1] .= dynmat
    checkpoint_idx = 2

    for iter in 1:n_iters
        # Generate configurations with current set of IFCs
        generate_configs!(configs, N_atoms, freqs, phi, z, eq_positions,
             rtm_idxs, atom_masses, temp, kB, hbar, mode; nthreads = nthreads)
        
        # Calculate tempearture dependent IFCs from generated configurations
        bar = ProgressBar(eachcol(configs), printing_delay = 0.2)
        set_description(bar, "Calculating AvgIFCs, Iter: $iter")
        for config in bar
            fill!(dynmat, T(0.0))
            xs = [config[D*(i-1) + 1 : D*i] for i in 1:N_atoms]  #*allocates
            sys = SuperCellSystem(xs, atom_masses, box_sizes)
            avg_dynmat .+= dynamical_matrix!(dynmat, sys, pot, calc)
        end

        avg_dynmat ./= n_configs

        freqs_sq, phi = get_modes(avg_dynmat, D) #*allocates
        freqs .= sqrt.(Complex.(freqs_sq)) * freq_unit
        rtm_idxs = rigid_translation_modes(freqs, D)


        if imaginary_mode_present(freqs)
            return ImaginaryModeError("Imaginary mode present on iteration $(iter)")
        end

        if (iter % save_every == 0) && (checkpoint_idx <= (size(freq_checkpoints, 2) - 1))
            freq_checkpoints[:, checkpoint_idx] .= ustrip.(freqs)
            dynmat_checkpoints[:, :, checkpoint_idx] .= avg_dynmat
            checkpoint_idx += 1
            push!(checkpoint_idxs, iter)
        end

        # Reset random numbers
        randn!(z)

        # Reset configs for re-use
        if iter != n_iters
            fill!(configs, 0.0*L_unit)
        end
    end

    freq_checkpoints[:, checkpoint_idx] .= ustrip.(freqs)
    dynmat_checkpoints[:, :, checkpoint_idx] .= avg_dynmat
    push!(checkpoint_idxs, n_iters)

    return SelfConsistentConfigs(configs, freq_checkpoints, dynmat_checkpoints, checkpoint_idxs,
                                 kB, hbar, temp, n_iters)
end