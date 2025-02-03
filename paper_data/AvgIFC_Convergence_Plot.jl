using CairoMakie
using JLD2

sizes = [150,250,350,450,550]
n_seeds = 20
size_in_inches = (3, 2.25)
dpi = 300
size_in_pixels = size_in_inches .* dpi

# LJ PLOT
basepath = "Z:/emeitz/Data/IFC_Convergence/LJ_AvgIFC_Convergence_N50_N550"
# basepath = "Z:/emeitz/Data/IFC_Convergence/LJ_TDEP_Convergence_N50_N550"
temps = [10,80]

cv_all = zeros((length(temps), length(sizes)))
cv_se_all = zeros((length(temps), length(sizes)))

for (i,size) in enumerate(sizes)
    for (j,temp) in enumerate(temps)
        path = joinpath(basepath, "N$(size)", "T$(temp)", "cv_data_averaged.jld2")
        cv, cv_se = load(path, "TEP_cv_total_avg", "TEP_cv_total_std_err")
        cv_all[j,i] = cv
        cv_se_all[j,i] = cv_se
    end
end

kB = 1.987204118e-3
kB_wrong = 8.617333262145e-5

f = Figure(resolution = size_in_pixels);
ax = Axis(f[1,1], xlabel = "Number of Samples", ylabel = L"C_{V,U} / (Nk_{\text{B}})",
    ylabelsize = 40, xlabelsize = 40, yticklabelsize = 30, xticklabelsize = 30,
    xgridvisible = false, ygridvisible = false, xticksmirrored = true, yticksmirrored = true, 
    xticklabelpad = 4, xtickalign=1, ytickalign = 1, yticks = [0.3,0.35,0.4,0.45,0.5], xticks = [0,150,250,350,450,550])
xlims!(0,600)
ylims!(0.28,0.52)
s1 = scatter!(sizes, cv_all[1,:], markersize = 40, color = :blue, strokecolor = :black, strokewidth = 2)
s2 = scatter!(sizes, cv_all[2,:], markersize = 40, color = :red, strokecolor = :black, strokewidth = 2)
axislegend(ax, [s1,s2], ["AvgInst 10 K", "AvgInst 80 K"], position = :rb, labelsize = 35, framevisible = false)
save(joinpath(basepath, "LJ_AvgIFC_convergence.png"), f)


# axislegend(ax, [s1,s2], ["TDEP-SS 10 K", "TDEP-SS 80 K"], position = :rt, labelsize = 35, framevisible = false)
# save(joinpath(basepath, "LJ_TDEP_convergence.png"), f)

# SW PLOT
basepath = "Z:/emeitz/Data/IFC_Convergence/SW_AvgIFC_Convergence_N50_N550"
temps = [100,1300]

cv_all = zeros((length(temps), length(sizes)))
cv_se_all = zeros((length(temps), length(sizes)))

for (i,size) in enumerate(sizes)
    for (j,temp) in enumerate(temps)
        path = joinpath(basepath, "N$(size)", "T$(temp)", "cv_data_averaged.jld2")
        cv, cv_se = load(path, "TEP_cv_total_avg", "TEP_cv_total_std_err")
        cv_all[j,i] = cv
        cv_se_all[j,i] = cv_se
    end
end


f = Figure(resolution = size_in_pixels);
ax = Axis(f[1,1], xlabel = "Number of Samples", ylabel = L"C_{V,U}/ (Nk_{\text{B}})",
    ylabelsize = 40, xlabelsize = 40, yticklabelsize = 30, xticklabelsize = 30,
    xgridvisible = false, ygridvisible = false, xticksmirrored = true, yticksmirrored = true, 
    xticklabelpad = 4, xtickalign = 1, ytickalign = 1,
    yticks = [0.5,0.55,0.6], xticks = [0,150,250,350,450,550])
xlims!(0,600)
ylims!(0.43,0.63)

s1 = scatter!(sizes, cv_all[1,:], markersize = 40, color = :blue, strokecolor = :black, strokewidth = 2)
s2 = scatter!(sizes, cv_all[2,:], markersize = 40, color = :red, strokecolor = :black, strokewidth = 2)
axislegend(ax, [s1,s2], ["AvgInst 100 K", "AvgInst 1300 K"], position = :rb, labelsize = 35, framevisible = false)
save(joinpath(basepath, "SW_AvgIFC_convergence.png"),f)