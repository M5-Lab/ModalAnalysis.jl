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
#TODO: write data in another CPU thread??

function balanced_partition(arr, n)
    len = length(arr)
    size = len ÷ n
    remainder = len % n
    starts = [1 + ((i-1) * size) + min(i-1, remainder) for i in 1:n]
    ends = [i * size + min(i, remainder) for i in 1:n]
    [arr[starts[i]:ends[i]] for i in 1:n]
end

pot = LJ(3.4, 0.24037, 8.5);
base_path = "/home/emeitz/MD_data/NMA_LJ_FCC_4UC/"
# TEP_path = raw"C:\Users\ejmei\repos\ModalAnalysis.jl\examples\LJ_FCC_4UC\TEP.jld2"
TEP_path = raw"/home/emeitz/MD_data/NMA_LJ_FCC_4UC/TEP.jld2"
kB = ustrip(u"kcal * mol^-1 * K^-1", Unitful.k*Unitful.Na)

temps = [40,50,60,70,80] #already ran 10-30
n_seeds = 10

total_threads = Threads.nthreads()
gpu_ids = CUDA.devices()
threads_per_task = div(total_threads, length(gpu_ids))

param_combos = Iterators.product(temps, 1:n_seeds)
gpu_jobs = balanced_partition(collect(param_combos), length(gpu_ids))

@assert length(gpu_jobs) == length(gpu_ids)


@sync for (gpu_id, gpu_job) in enumerate(gpu_jobs)
    # New CPU Thread for Each GPU
    Threads.@spawn begin
        CUDA.device!(gpu_id)
        #Launch GPU Jobs in Serial
        for param_combo in gpu_job
            temp, seed = param_combo

            @info "Starting temperature $(temp), seed $(seed) on GPU $(gpu_id)"
            seed_path = joinpath(base_path,"$(temp)K/seed$(seed-1)")

            equilibrium_data_path = joinpath(seed_path, "equilibrium.atom")
            dump_path = joinpath(seed_path, "dump.atom")
            thermo_path = joinpath(seed_path,"thermo_data.txt")

            eq = LammpsDump(equilibrium_data_path);
            parse_timestep!(eq, 1)
            atom_masses = get_col(eq, "mass")

            ld = LammpsDump(dump_path);

            #Load Thermo Data & Masses
            pe_col = 3; T_col = 2
            thermo_data = readdlm(thermo_path, skipstart = 2);
            potential_eng_MD = thermo_data[:,pe_col]
            temps_MD = thermo_data[:, T_col]

            T_avg = mean(temps_MD)

            NMA(eq, ld, potential_eng_MD, atom_masses, seed_path, TEP_path, gpu_id)
            NM_postprocess(seed_path, dirname(TEP_path), kB, T_avg;
                nthreads = threads_per_task, average_identical_freqs = true)
        end
    end
end

temps = [10,20,30,40,50,60,70,80]
MD_cv_arr = zeros(length(temps))
MD_std_err_arr = zeros(length(temps))
TEP_cv_arr = zeros(length(temps))
TEP_std_err_arr = zeros(length(temps))
Threads.@threads for (i,temp) in collect(enumerate(temps))
    temp_path = joinpath(base_path,"$(temp)K")
    MD_cv_total_avg, MD_cv_std_err, TEP_cv_total_avg, TEP_cv_std_err = NMA_avg_seeds(temp_path, n_seeds)
    TEP_cv_arr[i] = TEP_cv_total_avg
    TEP_std_arr[i] = TEP_cv_std
    MD_cv_arr[i] = MD_cv_total_avg
    MD_std_arr[i] = MD_cv_std
end

f = Figure()
ax = Axis(f[1,1], xlabel = "Temperature [K]", ylabel = "Total Heat Capacity")
s1 = scatter!(temps_to_parse, MD_cv_arr)
s2 = scatter!(temps_to_parse, TEP_cv_arr)
errorbars!(temps_to_parse, MD_cv_arr, MD_std_err_arr, MD_std_err_arr, whiskerwidth = 3, direction = :y)
errorbars!(temps_to_parse, TEP_cv_arr, TEP_std_err_arr, TEP_std_err_arr, whiskerwidth = 3, direction = :y)
Legend(f[1,2], [s1,s2], ["MD", "TEP"], "Energy Calculator")
save(joinpath(basepath,"HeatCap_vs_Temp.svg"), f)



###############
# NMA EXAMPLE #
###############
# base_path = raw"C:\Users\ejmei\repos\ModalAnalysis.jl\examples\LJ_FCC_4UC\NMA"
# base_path = raw"C:\Users\ejmei\Desktop\seed4"
# base_path = raw"C:\Users\ejmei\Desktop\WTF\Fresh Run"
# equilibrium_data_path = joinpath(base_path, "equilibrium.atom")
# dump_path = joinpath(base_path, "dump.atom")
# thermo_path = joinpath(base_path,"thermo_data.txt")

# eq = LammpsDump(equilibrium_data_path);
# parse_timestep!(eq, 1)
# atom_masses = get_col(eq, "mass")

# ld = LammpsDump(dump_path);

# #Load Thermo Data & Masses
# pe_col = 3; T_col = 2
# thermo_data = readdlm(thermo_path, skipstart = 2);
# potential_eng_MD = thermo_data[:,pe_col]
# temps = thermo_data[:, T_col]

# #Replicate potential used in LAMMSP
# pot = LJ(3.4, 0.24037, 8.5);

# out_path = base_path

# T_des = 10.0
# T_avg = mean(temps)

# #Dump needs xu,yu,zu
# gpu_id = 0
# mcc_block_size = 256

# # TEP_path = raw"C:\Users\ejmei\repos\ModalAnalysis.jl\examples\LJ_FCC_4UC\K3_4UC_LJ.jld2"
# kB = ustrip(u"kcal * mol^-1 * K^-1", Unitful.k*Unitful.Na)

# NMA(eq, ld, pot, potential_eng_MD, atom_masses, out_path, gpu_id, mcc_block_size)
# NM_postprocess(base_path, base_path, kB, T_avg; average_identical_freqs = true)


# NMA(eq, ld, pot, potential_eng_MD, atom_masses, out_path, gpu_id)
###############
# INMA EXAMPLE #
###############
# max_deviation = 0.01*mean(potential_eng_MD)

#Dump needs x, y, z, ix, iy, iz, fx, fy, fz
# INMA(ld, pot, potential_eng_MD, masses, max_deviation, out_path, T_des, kB)    