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
                                n_iters::Int, outpath::Function; save_every = 25,
                                mode = :quantum, nthreads = Threads.nthreads())

    if mode ∉ [:quantum, :classical]
        error("mode must be :quantum or :classical")
    end

    # all_configs = Vector{Result{SelfConsistentConfigs, ImaginaryModeError}}(undef, length(temperatures))

    # @tasks for temp in temperatures
    for (i,temp) in enumerate(temperatures)
        # @set ntasks = nthreads
        @info "Calculating Configurations for $temp K"
        cfg = self_consistent_IFC_loop(sys_eq, calc, temp, pot, n_configs, n_iters, mode;
                                         save_every = save_every, nthreads = nthreads)

        if typeof(cfg) == ImaginaryModeError
            @warn cfg.msg
        else
            sch_data = cfg.result.value 
            #* CHANGE THE HARD CODED PATH USED FOR TESTING
            jldopen(outpath(ustrip(temp)), "w") do f
                f["configs"] = ustrip.(sch_data.configs)
                f["freq_checkpoints"] = ustrip.(sch_data.freq_checkpoints)
                f["dynmat_checkpoints"] = sch_data.dynmat_checkpoints
                f["n_iters"] = sch_data.n_iters
                f["temp"] = ustrip(temp)
            end
        end
    end

end