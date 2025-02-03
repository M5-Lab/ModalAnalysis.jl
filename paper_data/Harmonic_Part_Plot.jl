using JLD2
using CairoMakie
using DelimitedFiles
using StatsBase

base_path = "Z:/emeitz/Data/HeatCapStudies/HeatCapPaper/LJ_Harmonic_ProveJM_Wrong"
base_outpath = joinpath(@__DIR__,"LJ_PLOTS")


temps = [10,40,80]

## Heat cap vs Temp, MD & TEP (TDEP  0K)
MD_cv_arr = zeros(length(temps)); TEP_cv_arr = zeros(length(temps)); TEP_TDEP_cv_arr = zeros(length(temps)); TEP_AvgIFC_cv_arr = zeros(length(temps))
MD_se_arr = zeros(length(temps)); TEP_se_arr = zeros(length(temps)); TEP_TDEP_se_arr = zeros(length(temps)); TEP_AvgIFC_se_arr = zeros(length(temps))

for (i,temp) in enumerate(temps)
    IFC_0K_path = joinpath(base_path, "0K", "T$(temp)", "cv_data_averaged.jld2")
    TDEP_path = joinpath(base_path, "TDEP", "T$(temp)", "cv_data_averaged.jld2")
    AvgIFC_path = joinpath(base_path, "AvgIFC", "T$(temp)", "cv_data_averaged.jld2")

    TEP_cv, TEP_se = load(IFC_0K_path, "TEP_cv_total_avg", "TEP_cv_total_std_err")
    TDEP_cv, TDEP_se = load(TDEP_path, "TEP_cv_total_avg", "TEP_cv_total_std_err")
    MD_cv, MD_se, AvgIFC_cv, AvgIFC_se = load(AvgIFC_path, "cv_MD_total_avg", "MD_cv_total_std_err", "TEP_cv_total_avg", "TEP_cv_total_std_err")


    TEP_cv_arr[i] = TEP_cv
    TEP_se_arr[i] = TEP_se
    TEP_TDEP_cv_arr[i] = TDEP_cv
    TEP_TDEP_se_arr[i] = TDEP_se
    MD_cv_arr[i] = MD_cv
    MD_se_arr[i] = MD_se
    TEP_AvgIFC_cv_arr[i] = AvgIFC_cv
    TEP_AvgIFC_se_arr[i] = AvgIFC_se
end  

# writedlm(joinpath(avg_ifc_data_path, "cv_data_all.csv"), [MD_cv_arr TEP_cv_arr TEP_TDEP_cv_arr TEP_AvgIFC_cv_arr],",")

colors = ["#3a5ed6", "#fcba03", "#fc08f8"]
size_in_inches = (3, 2.25)
dpi = 300
size_in_pixels = size_in_inches .* dpi

f = Figure(resolution = size_in_pixels);
ax = Axis(f[1,1], xlabel = L"$T$ [K]", ylabel = L"C_{V,U} / (Nk_{\text{B}})",
    ylabelsize = 50, xlabelsize = 40, yticklabelsize = 30, xticklabelsize = 30, yticks = [0.3, 0.5, 0.7, 0.9, 1.1],
    xgridvisible = false, ygridvisible = false, xticksmirrored = true, yticksmirrored = true,
    xticklabelpad = 4, xtickalign=1, ytickalign = 1)

xlims!(0,85)
ylims!(0.2,1.12)
h = hlines!(0.5, 0, 1350, color = :black, linestyle = :dash, linewidth = 4)
# s1 = scatter!(temps, MD_cv_arr, markersize = 30, marker = :star6, color = "#1c3afc");
s2 = scatter!(temps, TEP_cv_arr, markersize = 30, color = RGBAf(0,0,0,0), strokewidth = 6, strokecolor = colors[1]);
s3 = scatter!(temps, TEP_TDEP_cv_arr, markersize = 30, color = RGBAf(0,0,0,0), strokewidth = 6, strokecolor = colors[2]);
s4 = scatter!(temps, TEP_AvgIFC_cv_arr, markersize = 30, color = RGBAf(0,0,0,0), strokewidth = 6, strokecolor = colors[3]);
s1 = scatter!(temps, MD_cv_arr, markersize = 50, marker = '*', color = :black, strokewidth = 2);


e1 = [MarkerElement(color = RGBAf(0,0,0,0), strokecolor = :black, marker = '*', markersize = 40, strokewidth = 2)]
e2 = [MarkerElement(points = Point2f[(0.25, 0.5)], color = RGBAf(0,0,0,0), strokecolor = colors[1], marker = :circle, markersize = 35, strokewidth = 6)]
e3 = [MarkerElement(points = Point2f[(0.25, 0.5)], color = RGBAf(0,0,0,0), strokecolor = colors[2], marker = :circle, markersize = 35, strokewidth = 6)]
e4 = [MarkerElement(points = Point2f[(0.25, 0.5)], color = RGBAf(0,0,0,0), strokecolor = colors[3], marker = :circle, markersize = 35, strokewidth = 6)]
eh = [LineElement(color = :black, linestyle = :dash, linewidth = 4, points = Point2f[(-0.5, 0.5), (1.0, 0.5)])]

axislegend(ax, [[e2,e3,e4],[eh,e1]],
            [["ZeroK", "TDEP-SS", "AvgInst"],["Dulong-Petit", "MD (Ground Truth)"]], [nothing, nothing],
            position = :cb, labelsize = 35, orientation = :vertical, framevisible = false, nbanks = 3, 
            labelhalign = :center)
            
save(joinpath(base_outpath,"HeatCapHarmonic_LJ.svg"), f)


base_path = "Z:/emeitz/Data/HeatCapStudies/HeatCapPaper/SW_Harmonic_ProveJM_Wrong"
base_outpath = joinpath(@__DIR__,"SW_PLOTS")


temps = [100,700,1300]

## Heat cap vs Temp, MD & TEP (TDEP  0K)
MD_cv_arr = zeros(length(temps)); TEP_cv_arr = zeros(length(temps)); TEP_TDEP_cv_arr = zeros(length(temps)); TEP_AvgIFC_cv_arr = zeros(length(temps))
MD_se_arr = zeros(length(temps)); TEP_se_arr = zeros(length(temps)); TEP_TDEP_se_arr = zeros(length(temps)); TEP_AvgIFC_se_arr = zeros(length(temps))

for (i,temp) in enumerate(temps)
    IFC_0K_path = joinpath(base_path, "0K", "T$(temp)", "cv_data_averaged.jld2")
    TDEP_path = joinpath(base_path,"TDEP", "T$(temp)", "cv_data_averaged.jld2")
    AvgIFC_path = joinpath(base_path, "AvgIFC", "T$(temp)", "cv_data_averaged.jld2")

    TEP_cv, TEP_se = load(IFC_0K_path, "TEP_cv_total_avg", "TEP_cv_total_std_err")
    TDEP_cv, TDEP_se = load(TDEP_path, "TEP_cv_total_avg", "TEP_cv_total_std_err")
    MD_cv, MD_se, AvgIFC_cv, AvgIFC_se = load(AvgIFC_path, "cv_MD_total_avg", "MD_cv_total_std_err", "TEP_cv_total_avg", "TEP_cv_total_std_err")


    TEP_cv_arr[i] = TEP_cv
    TEP_se_arr[i] = TEP_se
    TEP_TDEP_cv_arr[i] = TDEP_cv
    TEP_TDEP_se_arr[i] = TDEP_se
    MD_cv_arr[i] = MD_cv
    MD_se_arr[i] = MD_se
    TEP_AvgIFC_cv_arr[i] = AvgIFC_cv
    TEP_AvgIFC_se_arr[i] = AvgIFC_se
end  


colors = ["#3a5ed6", "#fcba03", "#fc08f8"]
size_in_inches = (3, 2.25)
dpi = 300
size_in_pixels = size_in_inches .* dpi

f = Figure(resolution = size_in_pixels);
ax = Axis(f[1,1], xlabel = L"$T$ [K]", ylabel = L"C_{V,U} / (Nk_{\text{B}})",
    ylabelsize = 40, xlabelsize = 40, yticklabelsize = 30, xticklabelsize = 30, yticks = [0.4, 0.5, 0.6, 0.7, 0.8],
    xgridvisible = false, ygridvisible = false, xticksmirrored = true, yticksmirrored = true,
    xticklabelpad = 4, xtickalign=1, ytickalign = 1)
ylims!(0.38,0.85)
h = hlines!(0.5, 0, 1350, color = :black, linestyle = :dash, linewidth = 4)
# s1 = scatter!(temps, MD_cv_arr, markersize = 30, marker = :star6, color = "#1c3afc");
s2 = scatter!(temps, TEP_cv_arr, markersize = 30, color = RGBAf(0,0,0,0), strokewidth = 6, strokecolor = colors[1]);
s3 = scatter!(temps, TEP_TDEP_cv_arr, markersize = 30, color = RGBAf(0,0,0,0), strokewidth = 6, strokecolor = colors[2]);
s4 = scatter!(temps, TEP_AvgIFC_cv_arr, markersize = 30, color = RGBAf(0,0,0,0), strokewidth = 6, strokecolor = colors[3]);
s1 = scatter!(temps, MD_cv_arr, markersize = 50, marker = '*', color = :black, strokewidth = 2);

axislegend(ax, [[e2,e3,e4],[eh,e1]],
            [["ZeroK", "TDEP-SS", "AvgInst"],["Dulong-Petit", "MD (Ground Truth)"]], [nothing, nothing],
            position = :cb, labelsize = 35, orientation = :vertical, framevisible = false, nbanks = 3, 
            labelhalign = :center)

save(joinpath(base_outpath,"HeatCapHarmonic_SW.svg"), f)