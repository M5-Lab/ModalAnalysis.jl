using JLD2
using CairoMakie
using DelimitedFiles
using StatsBase

base_path = joinpath(@__DIR__ , "DATA/LJ_NMA/Bulk")
base_outpath = joinpath(@__DIR__,"LJ_PLOTS")


temps = [10,20,30,40,50,60,70,80]

## Heat cap vs Temp, MD & TEP (TDEP  0K)
MD_cv_arr = zeros(length(temps)); TEP_cv_arr = zeros(length(temps)); TEP_TDEP_cv_arr = zeros(length(temps)); TEP_AvgIFC_cv_arr = zeros(length(temps))
MD_se_arr = zeros(length(temps)); TEP_se_arr = zeros(length(temps)); TEP_TDEP_se_arr = zeros(length(temps)); TEP_AvgIFC_se_arr = zeros(length(temps))
os_cv_arr = zeros(length(temps)); os_se_arr = zeros(length(temps))
alm_ss_arr = zeros(length(temps));
alm_ss_se_arr = zeros(length(temps));

for (i,temp) in enumerate(temps)
    IFC_0K_path = joinpath(base_path, "0K", "cv_data_averaged_$(temp)K.jld2")
    TDEP_path = joinpath(base_path, "TDEP_StepByStep", "cv_data_averaged_$(temp)K.jld2")
    AvgIFC_path = joinpath(base_path, "AVGIFC_CLEAN", "cv_data_averaged_$(temp)K.jld2")
    os_path = joinpath(base_path, "OneShot", "cv_data_averaged_$(temp)K.jld2")

    TEP_cv, TEP_se = load(IFC_0K_path, "TEP_cv_total_avg", "TEP_cv_total_std_err")
    TDEP_cv, TDEP_se = load(TDEP_path, "TEP_cv_total_avg", "TEP_cv_total_std_err")
    MD_cv, MD_se, AvgIFC_cv, AvgIFC_se = load(AvgIFC_path, "cv_MD_total_avg", "MD_cv_total_std_err", "TEP_cv_total_avg", "TEP_cv_total_std_err")
    os_cv, os_se, = load(os_path, "TEP_cv_total_avg", "TEP_cv_total_std_err")

    TEP_cv_arr[i] = TEP_cv
    TEP_se_arr[i] = TEP_se
    TEP_TDEP_cv_arr[i] = TDEP_cv
    TEP_TDEP_se_arr[i] = TDEP_se
    MD_cv_arr[i] = MD_cv
    MD_se_arr[i] = MD_se
    TEP_AvgIFC_cv_arr[i] = AvgIFC_cv
    TEP_AvgIFC_se_arr[i] = AvgIFC_se
    os_cv_arr[i] = os_cv
    os_se_arr[i] = os_se
end  

# writedlm(joinpath(avg_ifc_data_path, "cv_data_all.csv"), [MD_cv_arr TEP_cv_arr TEP_TDEP_cv_arr TEP_AvgIFC_cv_arr],",")

colors = ["#3a5ed6", "#fcba03", "#fc08f8", "#02c77b"]
size_in_inches = (3, 2.25)
dpi = 300
size_in_pixels = size_in_inches .* dpi

f = Figure(resolution = size_in_pixels);
ax = Axis(f[1,1], xlabel = L"$T$ [K]", ylabel = L"C_{V,U} / (Nk_{\text{B}})",
    ylabelsize = 40, xlabelsize = 40, yticklabelsize = 30, xticklabelsize = 30,
     yticks = [0.0, 0.2, 0.3, 0.4, 0.5], xgridvisible = false, ygridvisible = false,
     xticksmirrored = true, yticksmirrored = true, xticklabelpad = 4, xtickalign=1, ytickalign = 1)
xlims!(0,85)
ylims!(0.1,0.52)
h = hlines!(0.5, 0, 1350, color = :black, linestyle = :dash, linewidth = 4)
# s1 = scatter!(temps, MD_cv_arr, markersize = 30, marker = '*', color = :black, strokewidth = 1);
s2 = scatter!(temps, TEP_cv_arr, markersize = 35, color = RGBAf(0,0,0,0), strokewidth = 6, strokecolor = colors[1]);
s3 = scatter!(temps, TEP_TDEP_cv_arr, markersize = 35, color = RGBAf(0,0,0,0), strokewidth = 6, strokecolor = colors[2]);
s4 = scatter!(temps, TEP_AvgIFC_cv_arr, markersize = 35, color = RGBAf(0,0,0,0), strokewidth = 6, strokecolor = colors[3]);
s5 = scatter!(temps, os_cv_arr, markersize = 35, color = RGBAf(0,0,0,0), strokewidth = 6, strokecolor = colors[4]);
s1 = scatter!(temps, MD_cv_arr, markersize = 40, marker = '*', color = :black, strokewidth = 2);

e1 = [MarkerElement(color = RGBAf(0,0,0,0), strokecolor = :black, marker = '*', markersize = 40, strokewidth = 2)]
e2 = [MarkerElement(color = RGBAf(0,0,0,0), strokecolor = colors[1], marker = :circle, markersize = 35, strokewidth = 6)]
e3 = [MarkerElement(color = RGBAf(0,0,0,0), strokecolor = colors[2], marker = :circle, markersize = 35, strokewidth = 6)]
e4 = [MarkerElement(color = RGBAf(0,0,0,0), strokecolor = colors[3], marker = :circle, markersize = 35, strokewidth = 6)]
e5 = [MarkerElement(color = RGBAf(0,0,0,0), strokecolor = colors[4], marker = :circle, markersize = 35, strokewidth = 6)]
eh = [LineElement(color = :black, linestyle = :dash, linewidth = 4, points = Point2f[(-0.5, 0.5), (1.0, 0.5)])]

axislegend(ax, [[e2,e3,e5,e4],[eh,e1]],
            [["ZeroK", "TDEP-SS", "TDEP-OS", "AvgInst"],["Dulong-Petit", "MD (Ground Truth)"]], [nothing, nothing],
            position = :cb, labelsize = 35, orientation = :vertical, framevisible = false, nbanks = 4, 
            labelhalign = :center, colgap = 25, patchlabelgap = 12)

save(joinpath(base_outpath,"HeatCap_vs_Temp_ALL.svg"), f)


#################################################
f = Figure(resolution = size_in_pixels);
ax = Axis(f[1,1], xlabel = L"$T$ [K]", ylabel = L"C_{V,U} / (Nk_{\text{B}})",
    ylabelsize = 40, xlabelsize = 40, yticklabelsize = 30, xticklabelsize = 30,
     yticks = [0.0, 0.2, 0.3, 0.4, 0.5], xgridvisible = false, ygridvisible = false,
     xticksmirrored = true, yticksmirrored = true, xticklabelpad = 4, xtickalign=1, ytickalign = 1)
xlims!(0,85)
ylims!(0.1,0.52)
h = hlines!(0.5, 0, 1350, color = :black, linestyle = :dash, linewidth = 4)
# s1 = scatter!(temps, MD_cv_arr, markersize = 30, marker = '*', color = :black, strokewidth = 1);
s2 = scatter!(temps, TEP_cv_arr, markersize = 30, color = RGBAf(0,0,0,0), strokewidth = 6, strokecolor = colors[1]);
s3 = scatter!(temps, TEP_TDEP_cv_arr, markersize = 30, color = RGBAf(0,0,0,0), strokewidth = 6, strokecolor = colors[2]);
s1 = scatter!(temps, MD_cv_arr, markersize = 40, marker = '*', color = :black, strokewidth = 2);
s5 = scatter!(temps, os_cv_arr, markersize = 35, color = RGBAf(0,0,0,0), strokewidth = 6, strokecolor = colors[4]);


axislegend(ax, [[e2,e3, e5],[eh,e1]],
            [["ZeroK", "TDEP-SS", "TDEP-OS"],["Dulong-Petit", "MD (Ground Truth)"]], [nothing, nothing],
            position = :cb, labelsize = 35, orientation = :vertical, framevisible = false, nbanks = 2, 
            labelhalign = :center, colgap = 25, patchlabelgap = 12)
save(joinpath(base_outpath,"HeatCap_vs_Temp_NOAVGINM.svg"), f)

#################################################
f = Figure(resolution = size_in_pixels);
ax = Axis(f[1,1], xlabel = L"$T$ [K]", ylabel = L"C_{V,U} / (Nk_{\text{B}})",
    ylabelsize = 40, xlabelsize = 40, yticklabelsize = 30, xticklabelsize = 30,
     yticks = [0.0, 0.2, 0.3, 0.4, 0.5], xgridvisible = false, ygridvisible = false,
     xticksmirrored = true, yticksmirrored = true, xticklabelpad = 4, xtickalign=1, ytickalign = 1)
xlims!(0,85)
ylims!(0.1,0.52)
h = hlines!(0.5, 0, 1350, color = :black, linestyle = :dash, linewidth = 4)
# s1 = scatter!(temps, MD_cv_arr, markersize = 30, marker = '*', color = :black, strokewidth = 1);
s2 = scatter!(temps, TEP_cv_arr, markersize = 30, color = RGBAf(0,0,0,0), strokewidth = 6, strokecolor = colors[1]);
s3 = scatter!(temps, TEP_TDEP_cv_arr, markersize = 30, color = RGBAf(0,0,0,0), strokewidth = 6, strokecolor = colors[2]);
s1 = scatter!(temps, MD_cv_arr, markersize = 40, marker = '*', color = :black, strokewidth = 2);

axislegend(ax, [[e2,e3],[eh,e1]],
            [["ZeroK", "TDEP-SS"],["Dulong-Petit", "MD (Ground Truth)"]], [nothing, nothing],
            position = :cb, labelsize = 35, orientation = :vertical, framevisible = false, nbanks = 2, 
            labelhalign = :center, colgap = 25, patchlabelgap = 12)
save(joinpath(base_outpath,"HeatCap_vs_Temp_TDEPSS.svg"), f)

f = Figure(resolution = size_in_pixels);
ax = Axis(f[1,1], xlabel = L"$T$ [K]", ylabel = L"C_{V,U} / (Nk_{\text{B}})",
    ylabelsize = 40, xlabelsize = 40, yticklabelsize = 30, xticklabelsize = 30,
    yticks = [0.0, 0.2, 0.3, 0.4, 0.5], xgridvisible = false, ygridvisible = false,
    xticksmirrored = true, yticksmirrored = true, xticklabelpad = 4, xtickalign=1, ytickalign = 1)
xlims!(0,85)
ylims!(0.1,0.52)
h = hlines!(0.5, 0, 1350, color = :black, linestyle = :dash, linewidth = 4)
# s1 = scatter!(temps, MD_cv_arr, markersize = 30, marker = '*', color = :black, strokewidth = 1);
s2 = scatter!(temps, TEP_cv_arr, markersize = 30, color = RGBAf(0,0,0,0), strokewidth = 6, strokecolor = colors[1]);
s1 = scatter!(temps, MD_cv_arr, markersize = 40, marker = '*', color = :black, strokewidth = 2);

axislegend(ax, [[e2],[eh,e1]],
            [["ZeroK"],["Dulong-Petit", "MD (Ground Truth)"]], [nothing, nothing],
            position = :cb, labelsize = 35, orientation = :vertical, framevisible = false, nbanks = 2, 
            labelhalign = :center, colgap = 25, patchlabelgap = 12)
save(joinpath(base_outpath,"HeatCap_vs_Temp_0K.svg"), f)

f = Figure(resolution = size_in_pixels);
ax = Axis(f[1,1], xlabel = L"$T$ [K]", ylabel = L"C_{V,U} / (Nk_{\text{B}})",
    ylabelsize = 40, xlabelsize = 40, yticklabelsize = 30, xticklabelsize = 30,
     yticks = [0.0, 0.2, 0.3, 0.4, 0.5], xgridvisible = false, ygridvisible = false,
     xticksmirrored = true, yticksmirrored = true, xticklabelpad = 4, xtickalign=1, ytickalign = 1)
xlims!(0,85)
ylims!(0.1,0.52)
h = hlines!(0.5, 0, 1350, color = :black, linestyle = :dash, linewidth = 4)
s1 = scatter!(temps, MD_cv_arr, markersize = 40, marker = '*', color = :black, strokewidth = 2);
axislegend(ax, [eh,e1], ["Dulong-Petit", "MD (Ground Truth)"], position = :cb, 
            labelsize = 35, orientation = :horizontal, framevisible = false, nbanks = 1, 
            labelhalign = :center, colgap = 25, patchlabelgap = 12)
save(joinpath(base_outpath,"HeatCap_vs_Temp_MD.svg"), f)

