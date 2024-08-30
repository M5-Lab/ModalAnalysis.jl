export NMA_GPU_Jobs

function balanced_partition(arr, n)
    len = length(arr)
    size = div(len, n)
    remainder = len % n
    starts = [1 + ((i-1) * size) + min(i-1, remainder) for i in 1:n]
    ends = [i * size + min(i, remainder) for i in 1:n]
    [arr[starts[i]:ends[i]] for i in 1:n]
end



"""
Calculates modal heat capacity from molecular dynamics data (LAMMPS). The code assumes that the molecular
dynamics data was over a set of parameters and will parallelize the calculation across all available NVIDIA GPUs.

# Arguments: 
- `sim_folder::String`
    Path to base folder of the simulation data.  
- `TEP_folder::String`  
    Path to base folder containing the force constants  
- `temperatures::AbstractVector{<:Real}`  
    Temperature is a required parameter for NMA. This is a list of temperatures to run NMA on.  
- `sim_folder_name::Function ` 
    A function that returns a list of strings given a tempearture, the other parameters and the seed number (`func(temp, params..., seed)`). 
    This strings are joined to `sim_folder` when running NMA via `joinpath`. For example if your seed 0, 10K simulation was in the
    folder /sim_folder/T_10K/seed0, then calling `sim_folder(10, 0)` should return ['T_10K', 'seed0']. If there are other params,
    pass them between the temperature and seed e.g. `sim_folder(10, "N10", 0)`. Note seeds are generated as zero indexed.
- `tep_file_name::Function`
    A function that returns a string given a temeprature and the other parameters (`func(temp, params...)`).
    This string is appended to `TEP_folder` when running NMA. This could just be the same string for all parameter sets.
- `n_seeds::Integer`
    Number of seeds run for each set of parameters. Note that seeds are generated as 0 indexed internally.
- `pot::Potential`
    Potential used in the simulation
- `other_params_to_sweep::Dict{String, AbstractVector{<:Real}} = Dict()`
    Dictionary of (non-temperature) parameters to sweep. By default this is Dict().
- `gpu_ids = CUDA.devices()`
    List of GPU ids to run the NMA on. By default this is CUDA.devices().
- `ncores::Integer = Threads.nthreads()`
    Number of CPU cores to use. By default this is Threads.nthreads().
- `avg_identical_freqs::Bool = false`
    If true, the modal data is geneated with identical freuencies are averaged. By default this is false.
- `order::Int = 3`
    Order of the force constants to use. By default this is 3.
"""
function NMA_GPU_Jobs(sim_folder::String, TEP_folder::String, temperatures::AbstractVector{<:Real},
     sim_folder_name::Function, tep_file_name::Function, n_seeds::Integer, pot::Potential;
     other_params_to_sweep::Dict{String, <:AbstractVector{<:Real}} = Dict{String, AbstractVector{<:Real}}(),
     gpu_ids = CUDA.devices(), ncores = Threads.nthreads(), mcc_block_size::Union{Integer, Nothing} = nothing,
     avg_identical_freqs = false, order = 3)

    if energy_unit(pot) == u"eV"
        kB = ustrip(u"eV/K", Unitful.k)
    elseif energy_unit(pot) == u"kcal/mol"
        kB = ustrip(u"kcal * mol^-1 * K^-1", Unitful.k*Unitful.Na)
    else
        error("Unknown unit system")
    end

    n_gpus = length(gpu_ids)
    threads_per_task = div(ncores, n_gpus)

    param_combos = Iterators.product(temperatures, values(other_params_to_sweep)..., 1:n_seeds)
    gpu_jobs = balanced_partition(collect(param_combos), n_gpus)

    @assert length(gpu_jobs) == n_gpus


    @sync for (gpu_id, gpu_job) in enumerate(gpu_jobs)
        # New CPU Thread for Each GPU
        @async begin
            CUDA.device!(gpu_id-1)
            #Launch GPU Jobs in Serial
            for (temp, params..., seed) in gpu_job   
                @info "Starting temperature $(temp), seed $(seed), other paranms: $(params) on GPU $(gpu_id)"
                seed_path = joinpath(sim_folder, sim_folder_name(temp, params...,seed-1)...)
                TEP_path = joinpath(TEP_folder, tep_file_name(temp, params...))

                nma = NormalModeAnalysis(seed_path, pot, temp)
                ModalAnalysis.run(nma, TEP_path; order = order)
                NM_postprocess(nma, kB; nthreads = threads_per_task, average_identical_freqs = avg_identical_freqs)
                GC.gc()
            end
            nothing
        end
    end
    
    @info "Making plots"
    for (temp, params..., seed) in param_combos
        seed_path = joinpath(sim_folder, sim_folder_name(temp, params...,seed-1)...)
        make_plots(seed_path, temp; average_identical_freqs = avg_identical_freqs)
    end
    
    @info "Averaing Seeds"
    for (temp, p...) in Iterators.product(temperatures, values(other_params_to_sweep)...)
        folder_paths = sim_folder_name(temp, p..., 0)
        NMA_avg_seeds(joinpath(sim_folder, folder_paths[1:end-1]...), n_seeds, temp)
    end

end
