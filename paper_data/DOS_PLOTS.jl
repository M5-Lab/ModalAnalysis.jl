using JLD2
using CairoMakie
using DelimitedFiles
using StatsBase


basepath = raw"C:\Users\ejmei\Box\MyPapers\SolidModalHeatCap\Figs\DATA\DOS"
outpath_LJ = raw"C:\Users\ejmei\Box\MyPapers\SolidModalHeatCap\Figs\LJ_PLOTS/"
outpath_SW = raw"C:\Users\ejmei\Box\MyPapers\SolidModalHeatCap\Figs\SW_PLOTS/"

inv_cm_to_THz = 33.356

LJ_0K_DOS = readdlm(joinpath(basepath, "LJ_0K", "lj444.dos"), comments = true)
LJ_80K_TDEP_DOS = readdlm(joinpath(basepath, "LJ_TDEP_80K", "lj444.dos"), comments = true)
LJ_80K_Avg_DOS = readdlm(joinpath(basepath, "LJ_Avg_80K", "lj444.dos"), comments = true)
LJ_80K_SS_DOS = readdlm(joinpath(basepath, "LJ_SingleShot_80K", "lj444_ss.dos"), comments = true)

SW_0K_DOS = readdlm(joinpath(basepath, "SW_0K", "si222.dos"), comments = true)
SW_1300K_TDEP_DOS = readdlm(joinpath(basepath, "SW_TDEP_1300K", "si333.dos"), comments = true)
SW_1300K_Avg_DOS = readdlm(joinpath(basepath, "SW_Avg_1300K", "si333.dos"), comments = true)
SW_1300K_SS_DOS = readdlm(joinpath(basepath, "SW_SingleShot_1300K", "si333.dos"), comments = true)

size_in_inches = (3, 2.25)
dpi = 300
size_in_pixels = size_in_inches .* dpi

colors = ["#3a5ed6", "#fcba03", "#fc08f8", "#02c77b"]
f = Figure(resolution = size_in_pixels);
ax = Axis(f[1,1], xlabel = L"$\omega_n / 2\pi$ [THz]", ylabel = L"\text{Density of States [1/THz]}",
    ylabelsize = 35, xlabelsize = 35, yticklabelsize = 30, xticklabelsize = 30,
    xgridvisible = false, ygridvisible = false, xticks = [0,0.5,1.0,1.5,2.0,2.5], yticks = [0,0.04,0.08,0.12],
    xticksmirrored = true, yticksmirrored = true, xtickwidth = 1.5, ytickwidth = 1.5, xticklabelpad = 4, xtickalign=1, ytickalign = 1)
xlims!(0,2.8)
ylims!(0,0.15)

l1 = lines!(LJ_0K_DOS[:,1] ./ inv_cm_to_THz, LJ_0K_DOS[:,2], markersize = 30, color = colors[1], linewidth = 5)
l2 = lines!(LJ_80K_TDEP_DOS[:,1] ./ inv_cm_to_THz, LJ_80K_TDEP_DOS[:,2], markersize = 30, color = colors[2], linewidth = 5)
l3 = lines!(LJ_80K_Avg_DOS[:,1] ./ inv_cm_to_THz, LJ_80K_Avg_DOS[:,2], markersize = 30, color = colors[3], linewidth = 5)
l4 = lines!(LJ_80K_SS_DOS[:,1] ./ inv_cm_to_THz, LJ_80K_SS_DOS[:,2], markersize = 30, color = colors[4], linewidth = 5)
axislegend(ax, [l1,l2,l4,l3], ["ZeroK", "TDEP-SS 80K", "TDEP-OS 80K", "AvgInst 80K"], position = :lt, labelsize = 35,
             orientation = :horizontal, framevisible = false, nbanks = 4)
save(joinpath(outpath_LJ, "LJ_DOS.svg"), f)
save(joinpath(outpath_LJ, "LJ_DOS.png"), f)


f = Figure(resolution = size_in_pixels);
ax = Axis(f[1,1], xlabel = L"$\omega_n / 2\pi$ [THz]", ylabel = L"\text{Density of States [1/THz]}",
    ylabelsize = 35, xlabelsize = 35, yticklabelsize = 30, xticklabelsize = 30,
    xgridvisible = false, ygridvisible = false, xticks = [0,4,8,12,16], yticks = [0.0, 0.02, 0.04, 0.06],
    xticksmirrored = true, yticksmirrored = true, xticklabelpad = 4, xtickalign=1, ytickalign = 1)
xlims!(0,18.5)
ylims!(0, 0.065)

l1 = lines!(SW_0K_DOS[:,1] ./ inv_cm_to_THz, SW_0K_DOS[:,2], markersize = 30, color = colors[1], linewidth = 5)
l2 = lines!(SW_1300K_TDEP_DOS[:,1] ./ inv_cm_to_THz, SW_1300K_TDEP_DOS[:,2], markersize = 30, color = colors[2], linewidth = 5)
l3 = lines!(SW_1300K_Avg_DOS[:,1] ./ inv_cm_to_THz, SW_1300K_Avg_DOS[:,2], markersize = 30, color = colors[3], linewidth = 5)
l4 = lines!(SW_1300K_SS_DOS[:,1] ./ inv_cm_to_THz, SW_1300K_SS_DOS[:,2], markersize = 30, color = colors[4],
             linewidth = 5, linestyle = :dash)
axislegend(ax, [l1,l2,l4,l3], ["ZeroK", "TDEP-SS 1300K", "TDEP-OS 1300K", "AvgInst 1300K"], position = :lt, labelsize = 35, framevisible = false,
            orientation = :horizontal, nbanks = 4)
save(joinpath(outpath_SW, "SW_DOS.svg"), f)