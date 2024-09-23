export SelfConsistentLoopJob

###
# 1. Calculate 0K force constants
# 2. Generate atomic configurations
# 3. Calculate tempearture dependent IFCs from the configrautions in 2.
# 4. Calculate convergence metric
# 5. Repeat 2-4 until convergence/maxiters is reached


"""
    self_consistent_IFC_loop(temp::Real, pot::Potential, outpath::String; mode = :quantum)

Perform a self-consistent loop to calculate temperature dependent IFCs.

# Arguments
- `sys_eq::SuperCellSystem`: Equilibrium system to initialize calculation.
- `temperatures::AbstractVector{<:Real}`: Temperatures to calculate IFCs at.
- `pot::Potential`: Potential to calculate IFCs with.
- `outpath::String`: Path to save output files.
- `mode::Symbol`: Statistics to use, `:quantum` or `:classical`.
"""
function SelfConsistentLoopJob(sys_eq::SuperCellSystem, temperatures::AbstractVector,
                                calc::ForceConstantCalculator, pot::Potential, n_configs::Int,
                                n_iters::Int, outpath::String;  mode = :quantum, ncores = Threads.nthreads())

    if mode ∉ [:quantum, :classical]
        error("mode must be :quantum or :classical")
    end

    all_configs = Vector{Result{SelfConsistentConfigs, ImaginaryModeError}}(undef, length(temperatures))

    # @tasks for temp in temperatures
    for (i,temp) in enumerate(temperatures)
        # @set ntasks = ncores
        @info "Calculating Configurations for $temp K"
        all_configs[i] = self_consistent_IFC_loop(sys_eq, calc, temp, pot, n_configs, n_iters, mode)

        if typeof(all_configs[i]) == ImaginaryModeError
            @warn all_configs[i].msg
        end
    end

end