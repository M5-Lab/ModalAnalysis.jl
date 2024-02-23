export MonteCarloMAJob


"""
Assumes positions given in the system are the reference positions to calculate displacements from. 

Generates output that is compatible with `NMA`` or `INMA`.

Currently no parallelization within a simulation, run as many seeds as cores available. Each seed
creates 0.5 GB of position storage plus other storage. Ensure you have ~ 1GB of RAM per seed plus
enough to store the second and third order force constants in RAM.
"""
function MonteCarloMAJob(sys::SuperCellSystem{D}, TEP_path::Function,
     n_steps::Int, n_steps_equil::Int, outpath::String,
     temperatures::AbstractVector{<:Real}, kB, data_interval::Int, n_seeds::Int, length_scale; output_type = :NMA,
     F2_name::String = "F2", F3_name::String = "F3") where D

     mkdir(outpath)

    if output_type ∉ (:NMA, :INMA)
        error("output_type must be :NMA or :INMA got $(output_type)")
    end

    if Threads.nthreads() < length(temperatures)
        error("Cannot have more temperatures: $(length(temperatures)) than threads: $(Threads.nthreads()).")
    end

    cores_per_temp = floor(Threads.nthreads() / length(temperatures))

    #Pick step sizes based on temperatures
    step_size_stds = zeros(length(temperatures))
    Threads.@threads for (i,temp) in enumerate(temperatures)
        step_size_stds[i], percent_accepted = pick_step_size(sys, 10000, length_scale, F2, F3, temp, kB)
        @info "Chose step size std: $(step_size_std) for temperature: $(temp)K with $(percent_accepted)% accepted."
    end

    Threads.@threads for (i,temp) in enumerate(temperatures)
        F2, F3 =  load(TEP_path(temp), F2_name, F3_name)

        Threads.@threads for chunk_i in 1:cores_per_temp
            for seed in chunk_i:cores_per_temp:n_seeds

                @info "Starting seed $(seed) for temperature $(temp)K"
                outpath_seed = joinpath(outpath, "seed$(seed)")
                mkdir(outpath_seed)

                sim = MC_Simulation(n_steps, n_steps_equil, step_size_std[i], temp, kB, deepcopy(positions(sys)))

                U_arr, num_accepted = runMC(sys, sim, outpath_seed, output_type, data_interval, F2, F3)

                jldsave(joinpath(outpath_seed, "MC_stats.jld2"), 
                        U_arr = U_arr, percent_accepted = 100*num_accepted/sim.n_steps,
                        step_size_std = step_size_std)
            end
            
        end
    end

end


function MonteCarloJob()
    #TODO just runs with empirical potential like LJ
end

