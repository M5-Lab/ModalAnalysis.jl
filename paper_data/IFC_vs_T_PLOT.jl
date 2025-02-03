using JLD2
using CairoMakie

function parse_IFC_vs_T(temps, m, tdep_ss_dir, tdep_os_dir, avgifc_dir, tdep_ss_filename,
                            tdep_os_filename, avgifc_filename, zerok_path, idx = [1,1])

    d_0K = load(zerok_path, "dynmat")

    val_0K = d_0K[idx[1],idx[2]]*m

    avgifc_vals = zeros(length(temps))
    tdep_ss_vals = zeros(length(temps))
    tdep_os_vals = zeros(length(temps))
    for (i,temp) in enumerate(temps)
        d_avgifc = load(joinpath(avgifc_dir, avgifc_filename(temp)), "dynmat")
        d_tdep_ss = load(joinpath(tdep_ss_dir, tdep_ss_filename(temp)), "dynmat")
        d_tdep_os = load(joinpath(tdep_os_dir, tdep_os_filename(temp)), "dynmat")

        avgifc_vals[i] = d_avgifc[idx[1],idx[2]]*m
        tdep_ss_vals[i] = d_tdep_ss[idx[1],idx[2]]*m
        tdep_os_vals[i] = d_tdep_os[idx[1],idx[2]]*m
    end

    return avgifc_vals, tdep_ss_vals, tdep_os_vals, val_0K
end

base_outpath = "C:/Users/ejmei/Box/MyPapers/SolidModalHeatCap/Figs/OTHER_PLOTS"

# LJ
m = 39.95
temps = [10,20,30,40,50,60,70,80]
avgifc_filename = (temp) -> "AvgIFC_LJ_$(temp)K_CLEANED.jld2"
tdep_ss_filename = (temp) -> "LJ_$(temp)K_residual.jld2"
tdep_os_filename = (temp) -> "LJ_$(temp)K_singleshot.jld2"

avgifc_dir = "Z:/emeitz/Data/ForceConstants/AvgINM_LJ"
tdep_ss_dir = "Z:/emeitz/Data/ForceConstants/LJ_ALM"
tdep_os_dir = "Z:/emeitz/Data/ForceConstants/LJ_ALM"
zerok_path = "Z:/emeitz/Data/NMA/LJ/LJ_FCC_Qual/IFC_0K/TEP1040.jld2"

avgifc_vals, tdep_ss_vals, tdep_os_vals, val_0K = parse_IFC_vs_T(temps, m, tdep_ss_dir, tdep_os_dir,
                                                     avgifc_dir, tdep_ss_filename, tdep_os_filename, avgifc_filename, zerok_path)

size_in_inches = (3, 2.25)
dpi = 300
size_in_pixels = size_in_inches .* dpi
f = Figure(resolution = size_in_pixels);
ax = Axis(f[1,1], xlabel = L"$T$ [K]", ylabel = L"\Phi_{11}^{\text{xx}} \:\: \text{[kcal/(mol*Å²)]}" ,
    ylabelsize = 40, xlabelsize = 40, yticklabelsize = 30, xticklabelsize = 30,
    xticks = [0, 10, 20, 30, 40, 50, 60, 70, 80], xgridvisible = false, ygridvisible = false, xticksmirrored = true, yticksmirrored = true,
    xticklabelpad = 4, xtickalign=1, ytickalign = 1)

#  ["#3a5ed6", "#fcba03", "#fc08f8", "#02c77b"]
s1 = scatter!(temps, avgifc_vals, color = "#fc08f8", markersize = 30, label = "AvgInst", strokewidth = 1)
s2 = scatter!(temps, tdep_ss_vals, color = "#fcba03", markersize = 30, label = "TDEP-SS", strokewidth = 1)
s4 = scatter!(temps, tdep_os_vals, color = "#02c77b", markersize = 30, label = "TDEP-OS", strokewidth = 1)
s3 = scatter!([0], [val_0K], color = "#3a5ed6", markersize = 30, label = "ZeroK", strokewidth = 1)

axislegend(ax, [s1,s2,s4,s3], ["AvgInst", "TDEP-SS", "TDEP-OS", "ZeroK"], position = :lt, labelsize = 35, framevisible = false)
save(joinpath(base_outpath,"IFCvT_LJ.png"), f)


# SW
m = 28.085
temps = [100,300,500,700,900,1100,1300]
avgifc_filename = (temp) -> "AvgIFC_SW_3UC_$(temp)K_CLEANED.jld2"
tdep_ss_filename = (temp) -> "SW_$(temp)K_residual.jld2"
tdep_os_filename = (temp) -> "SW_$(temp)K_singleshot.jld2"

tdep_ss_dir = "Z:/emeitz/Data/ForceConstants/SW_StepByStep"
avgifc_dir = "Z:/emeitz/Data/ForceConstants/AvgINM_SW"
tdep_os_dir = "Z:/emeitz/Data/ForceConstants/SW_OneShot"
zerok_path = "Z:/emeitz/Data/ForceConstants/SW_3UC_0K.jld2"

avgifc_vals, tdep_ss_vals, tdep_os_vals, val_0K = parse_IFC_vs_T(temps, m, tdep_ss_dir, tdep_os_dir,
                                                     avgifc_dir, tdep_ss_filename, tdep_os_filename, avgifc_filename, zerok_path)

size_in_inches = (3, 2.25)
dpi = 300
size_in_pixels = size_in_inches .* dpi
f = Figure(resolution = size_in_pixels);
ax = Axis(f[1,1], xlabel = L"$T$ [K]", ylabel = L"\Phi_{11}^{\text{xx}} \:\: \text{[eV/Å²]}",
    ylabelsize = 40, xlabelsize = 40, yticklabelsize = 30, xticklabelsize = 30,
    xticks = [0,100,300,500,700,900,1100,1300], xgridvisible = false, ygridvisible = false, xticksmirrored = true, yticksmirrored = true,
    xticklabelpad = 4, xtickalign=1, ytickalign = 1)

s1 = scatter!(temps, avgifc_vals, color = "#fc08f8", markersize = 30, label = "AvgInst", strokewidth = 1)
s2 = scatter!(temps, tdep_ss_vals, color = "#fcba03", markersize = 30, label = "TDEP-SS", strokewidth = 1)
s4 = scatter!(temps, tdep_os_vals, color = "#02c77b", markersize = 30, label = "TDEP-OS", strokewidth = 1)
s3 = scatter!([0], [val_0K], color = "#3a5ed6", markersize = 30, label = "ZeroK", strokewidth = 1)


axislegend(ax, [s1,s2,s4,s3], ["AvgInst", "TDEP-SS", "TDEP-OS", "ZeroK"], position = :rt, labelsize = 35, framevisible = false)
save(joinpath(base_outpath,"IFCvT_SW.png"), f)