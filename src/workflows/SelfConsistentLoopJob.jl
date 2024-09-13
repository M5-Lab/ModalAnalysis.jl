

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
function SelfConsistentLoopJob(sys_eq::SuperCellSystem, temperatures::AbstractVector{<:Real},
                                pot::Potential, outpath::String;
                                mode = :quantum, ncores = Threads.nthreads())

    if mode ∉ [:quantum, :classical]
        error("mode must be :quantum or :classical")
    end

    @tasks for temp in temperatures
        @set ntasks = ncores
        self_consistent_IFC_loop(sys_eq, temp, pot, outpath; mode = mode)
    end

end