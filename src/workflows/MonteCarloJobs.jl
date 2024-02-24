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

    if output_type ∉ (:NMA, :INMA, :None)
        error("output_type must be :NMA or :INMA got $(output_type)")
    end

    if Threads.nthreads() < length(temperatures)
        error("Cannot have more temperatures: $(length(temperatures)) than threads: $(Threads.nthreads()).")
    end

    cores_per_temp = floor(Int64, Threads.nthreads() / length(temperatures))

    #Pick step sizes based on temperatures
    step_size_stds = zeros(length(temperatures))

    Threads.@threads for (i,temp) in collect(enumerate(temperatures))
        f = jldopen(TEP_path(temp), "r"; parallel_read = true)
        F2 = f[F2_name]; F3 = f[F3_name]
        close(f)

        step_size_stds[i], percent_accepted = pick_step_size(sys, 10000, length_scale, F2, F3, temp, kB)
        @info "Chose step size std: $(step_size_stds[i]) for temperature: $(temp)K with $(percent_accepted)% accepted."

        Threads.@threads for chunk_i in 1:cores_per_temp
            for seed in chunk_i:cores_per_temp:n_seeds

                @info "Starting seed $(seed) for temperature $(temp)K"
                outpath_seed = joinpath(outpath, "T$(temp)", "seed$(seed)")
                mkpath(outpath_seed)

                sim = MC_Simulation(n_steps, n_steps_equil, step_size_stds[i], temp, kB, deepcopy(positions(sys)))

                U_arr, num_accepted = runMC(sys, sim, outpath_seed, output_type, data_interval, F2, F3)

                jldsave(joinpath(outpath_seed, "MC_stats.jld2"), 
                    U_arr = U_arr,
                    percent_accepted = 100*num_accepted/sim.n_steps,
                    step_size_std = step_size_stds[i],
                    cv_norm = var(U_arr)/((kB^2)*(temp^2)*(3*(n_atoms(sys)- 1))),
                    kB = kB,
                    T = temp,
                    ndof = 3*(n_atoms(sys)- 1)
                )

                @info "Finished seed $(seed) for temperature $(temp)K with $(100*num_accepted/sim.n_steps)% accepted."
            end
            
        end

        @info "Averaging Seeds"
        Threads.@threads for (i,temp) in collect(enumerate(temperatures))
            cvs_T = zeros(n_seeds)
            for seed in 1:n_seeds
                data_path = joinpath(outpath, "T$(temp)", "seed$(seed)", "MC_stats.jld2")
                cvs_T[seed] = load(data_path, "cv_norm")
            end

            cv_mean = mean(cvs_T)
            std_err = std(cvs_T)/sqrt(n_seeds)
            jldsave(joinpath(outpath, "T$(temp)", "cv_averaged.jld2"), cv_norm_avg = cv_mean, cv_norm_std_err = std_err)
        end

    end

end


function MonteCarloJob()
    #TODO just runs with empirical potential like LJ
end

