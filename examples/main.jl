using Revise
using Pkg
Pkg.activate(raw"C:\Users\ejmei\repos\ModalAnalysis.jl")
Pkg.instantiate()
Pkg.resolve()

using ModalAnalysis
using Unitful
using ForceConstants
using StatsBase
using DelimitedFiles
import CUDA

#TODO: get it to handle units

function balanced_partition(arr, n)
    len = length(arr)
    size = div(len, n)
    remainder = len % n
    starts = [1 + ((i-1) * size) + min(i-1, remainder) for i in 1:n]
    ends = [i * size + min(i, remainder) for i in 1:n]
    [arr[starts[i]:ends[i]] for i in 1:n]
end

pot = LJ(3.4, 0.24037, 8.5);
base_path = raw"C:\Users\ejmei\Desktop\WTF\Fresh Run"
# base_path = "/home/emeitz/MD_data/NMA_LJ_FCC_4UC/"
TEP_path = raw"C:\Users\ejmei\repos\ModalAnalysis.jl\examples\LJ_FCC_4UC\TEP.jld2"
# TEP_path = raw"/home/emeitz/MD_data/NMA_LJ_FCC_4UC/TEP.jld2"
kB = ustrip(u"kcal * mol^-1 * K^-1", Unitful.k*Unitful.Na)

# temps = [10,20,30,40,50,60,70,80]
temps = [50,60,70,80]
n_seeds = 10

total_threads = Threads.nthreads()
gpu_ids = [0]#CUDA.devices()
threads_per_task = div(total_threads, length(gpu_ids))

param_combos = Iterators.product(temps, 1:n_seeds)
gpu_jobs = balanced_partition(collect(param_combos), length(gpu_ids))

@assert length(gpu_jobs) == length(gpu_ids)


@sync for (gpu_id, gpu_job) in enumerate(gpu_jobs)
    # New CPU Thread for Each GPU
    @async begin #Threads.@spawn prob better but breaks
        CUDA.device!(gpu_id-1)
        #Launch GPU Jobs in Serial
        for param_combo in gpu_job
            temp, seed = param_combo

            @info "Starting temperature $(temp), seed $(seed) on GPU $(gpu_id)"
            seed_path = joinpath(base_path,"$(temp)K/seed$(seed-1)")

            nma = NormalModeAnalysis(seed_path, pot, temp)

            ModalAnalysis.run(nma, TEP_path)

            NM_postprocess(nma, kB; nthreads = threads_per_task, average_identical_freqs = true)
        end
        nothing
    end
end

for param_combo in param_combos
    temp, seed = param_combo
    seed_path = joinpath(base_path,"$(temp)K/seed$(seed-1)")
    make_plots(seed_path, temp)
end


for (i,temp) in collect(enumerate(temps))
    NMA_avg_seeds(joinpath(base_path, "$(temp)K"), n_seeds, temp)
end

f = Figure()
ax = Axis(f[1,1], xlabel = "Temperature [K]", ylabel = "Total Heat Capacity")
s1 = scatter!(temps_to_parse, MD_cv_arr)
s2 = scatter!(temps_to_parse, TEP_cv_arr)
errorbars!(temps_to_parse, MD_cv_arr, MD_std_err_arr, MD_std_err_arr, whiskerwidth = 3, direction = :y)
errorbars!(temps_to_parse, TEP_cv_arr, TEP_std_err_arr, TEP_std_err_arr, whiskerwidth = 3, direction = :y)
Legend(f[1,2], [s1,s2], ["MD", "TEP"], "Energy Calculator")
save(joinpath(basepath,"HeatCap_vs_Temp.svg"), f)