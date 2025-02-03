using JLD2
using CairoMakie
using DelimitedFiles
using StatsBase
using LinearAlgebra

# pot = "LJ"
# dos_temp = 80
# mode_temps = [10, 40, 80]

pot = "SW"
dos_temp = 1300
mode_temps = [100, 700, 1300]

outpath = raw"C:\Users\ejmei\Box\MyPapers\SolidModalHeatCap\Figs\COMBINED_PLOTS"

function DOS_PLOT(ax, pot, temp)
    colors = ["#3a5ed6", "#fcba03", "#fc08f8", "#02c77b"]
    basepath_dos = raw"C:\Users\ejmei\Box\MyPapers\SolidModalHeatCap\Figs\DATA\DOS"
    inv_cm_to_THz = 33.356
    

    if pot == "LJ"
        ZeroK_DOS = readdlm(joinpath(basepath_dos, "LJ_0K", "lj444.dos"), comments = true)
        TDEP_DOS = readdlm(joinpath(basepath_dos, "LJ_TDEP_80K", "lj444.dos"), comments = true)
        Avg_DOS = readdlm(joinpath(basepath_dos, "LJ_Avg_80K", "lj444.dos"), comments = true)
        SS_DOS = readdlm(joinpath(basepath_dos, "LJ_SingleShot_80K", "lj444_ss.dos"), comments = true)
        ss_linestyle = nothing
    elseif pot == "SW"
        ZeroK_DOS = readdlm(joinpath(basepath_dos, "SW_0K", "si222.dos"), comments = true)
        TDEP_DOS = readdlm(joinpath(basepath_dos, "SW_TDEP_1300K", "si333.dos"), comments = true)
        Avg_DOS = readdlm(joinpath(basepath_dos, "SW_Avg_1300K", "si333.dos"), comments = true)
        SS_DOS = readdlm(joinpath(basepath_dos, "SW_SingleShot_1300K", "si333.dos"), comments = true)
        ss_linestyle = :dash
    else
        error("Invalid potential")
    end
    
    l1 = lines!(ax, ZeroK_DOS[:,1] ./ inv_cm_to_THz, ZeroK_DOS[:,2], markersize = 30, color = colors[1], linewidth = 5)
    l2 = lines!(ax, TDEP_DOS[:,1] ./ inv_cm_to_THz, TDEP_DOS[:,2], markersize = 30, color = colors[2], linewidth = 5)
    l3 = lines!(ax, Avg_DOS[:,1] ./ inv_cm_to_THz, Avg_DOS[:,2], markersize = 30, color = colors[3], linewidth = 5)
    l4 = lines!(ax, SS_DOS[:,1] ./ inv_cm_to_THz, SS_DOS[:,2], markersize = 30, color = colors[4], linewidth = 5,
                linestyle = ss_linestyle)
    axislegend(ax, [l1,l2,l4,l3], ["ZeroK", "TDEP-SS $(temp)K", "TDEP-OS $(temp)K", "AvgInst $(temp)K"], position = (0.0,0.8), labelsize = 35,
                 orientation = :horizontal, framevisible = false, nbanks = 4)
end

function mode_cv_plot(ax, pot, xmax, temps)
    plts = []
    data_path = joinpath("C:/Users/ejmei/Box/MyPapers/SolidModalHeatCap/Figs/DATA","$(pot)_NMA/Modal")
    colors = ["#a3f55b","#1dd6f2", "#f21b6a"]
    legend_labels = ["AvgInst $(temp) K" for temp in temps]

    if pot == "LJ"
        conv = sqrt(418.4)/(2*pi)
    elseif pot == "SW"
        conv = sqrt(9.64897e3)/(2*pi)
    else
        error("Invalid potential")
    end

    h = hlines!(ax, 0.5, 0, xmax, color = :black, linestyle = :dash, linewidth = 4)
    eh = [LineElement(color = :black, linestyle = :dash, linewidth = 4, points = Point2f[(-0.5, 0.5), (1.0, 0.5)])]

    # axislegend(ax, [eh], ["Dulong-Petit"], position = :rt, labelsize = 35, orientation = :horizontal, framevisible = false)

    for (i,temp) in enumerate(temps)

        freqs_path = joinpath(data_path, "AVGIFC_CLEAN", "cv_data_averaged_$(temp)K.jld2")
        freqs_unique, cv_data_avg, cv_se = load(freqs_path, "unique_freqs", "TEP_cv_per_mode_avg", "TEP_cv_per_mode_std_err")

        errorbars!(ax, conv.*freqs_unique, cv_data_avg, cv_se, cv_se, whiskerwidth = 10, direction = :y, color = :black)
        push!(plts, scatter!(ax, conv.*freqs_unique, cv_data_avg, strokewidth = 1, color = colors[i], markersize = 25));

    end

    push!(plts, eh)
    leg = axislegend(ax, plts, [legend_labels; "Dulong-Petit"], position = :rt, 
                    labelsize = 35, framevisible = false, patchsize = (25.0f0,25.0f0))

end

function cov_plot(ax, pot, xmax)

    n_seeds = 50
    if pot == "LJ"
        conv = sqrt(418.4)/(2*pi)
        base_path = "Z:/emeitz/Data/NMA/LJ/LJ_NMA_Langevin_CLEAN"
        kB = 1.987204e-3
        n_modes = 256*3
        cov_temps = [10, 80]
    elseif pot == "SW"
        conv = sqrt(9.64897e3)/(2*pi)
        base_path = "Z:/emeitz/Data/NMA/SW/SW_3UC_NMA_CLEANAvgIFC_LANGEVIN"
        kB = 8.617333e-5
        n_modes = 216*3
        cov_temps = [100,1300]
    else
        error("Invalid potential")
    end

    

    ## Heat cap vs Temp, MD & TEP (TDEP  0K)
    all_cov = zeros(length(cov_temps), n_modes, n_seeds)
    all_var = zeros(length(cov_temps), n_modes, n_seeds)
    freqs = zeros(length(cov_temps), n_modes)
    
    for (i,temp) in enumerate(cov_temps)
        for s in range(1,n_seeds)
            path = joinpath(base_path, "T$(temp)", "seed$(s-1)", "cv_data.jld2")
            cov_matrix, freqs_local = load(path, "cv3_cov", "freqs")
            
            all_var[i,:,s] .= diag(cov_matrix)
            all_cov[i,:,s] .= (vec(sum(cov_matrix, dims = 2)) .- view(all_var, i, : , s))

            if s == 1
                freqs[i, :] = freqs_local.*conv
            end
        end
    end  

    all_cov ./= kB
    all_var ./= kB

    # AVERAGE BY FREQUENCY
    digits = 6
    freqs_rounded = [round.(freqs[i,:], digits = digits) for i in 1:length(cov_temps)]
    freqs_unique_all = [unique(freqs_rounded[i]) for i in 1:length(cov_temps)]
    var_part = [Float64[] for _ in 1:length(cov_temps)]
    cov_part = [Float64[] for _ in 1:length(cov_temps)]
    var_se = [Float64[] for _ in 1:length(cov_temps)]
    cov_se = [Float64[] for _ in 1:length(cov_temps)]

    for (i,T) in enumerate(cov_temps)
        var_part_unique = [Float64[] for _ in 1:length(freqs_unique_all[i])]
        cov_part_unique = [Float64[] for _ in 1:length(freqs_unique_all[i])]

        for (j, f) in enumerate(freqs_unique_all[i])
            @views idxs = findall(x -> x == f, freqs_rounded[i])
            
            append!(var_part_unique[j], collect(Iterators.flatten(all_var[i,idxs,:])))
            append!(cov_part_unique[j], collect(Iterators.flatten(all_cov[i,idxs,:])))
        end

        append!(var_part[i], mean.(var_part_unique))
        append!(cov_part[i], mean.(cov_part_unique))
        append!(var_se[i], std.(var_part_unique)./sqrt.(length.(var_part_unique)))
        append!(cov_se[i], std.(cov_part_unique)./sqrt.(length.(cov_part_unique)))
    end

    colors = ["#a3f55b", "#f21b6a"]


    h = hlines!(ax, 0.5, 0, xmax, color = :black, linestyle = :dash, linewidth = 4)

    for (i, T) in enumerate(cov_temps)

        s3 = errorbars!(ax, freqs_unique_all[i][4:end], var_part[i][4:end], var_se[i][4:end])
        s4 = errorbars!(ax, freqs_unique_all[i][4:end], cov_part[i][4:end], cov_se[i][4:end])
        s1 = scatter!(ax, freqs_unique_all[i][4:end], var_part[i][4:end], markersize = 25, 
                    color = colors[i], strokewidth = 1, marker = :diamond);
        s2 = scatter!(ax, freqs_unique_all[i][4:end], cov_part[i][4:end], markersize = 27, 
                    color = colors[i], strokewidth = 1, marker = :star6);
    end


    # el1 = [MarkerElement(color = RGBAf(0,0,0,0), marker = :diamond, markersize = 25, strokewidth = 2)]
    # el2 = [MarkerElement(points = [Point2f(0.5, 1.1)], marker = :star6, color = RGBAf(0,0,0,0), markersize = 27, strokewidth = 2)]
    el1 = [MarkerElement(points = [Point2f(-0.2, 0.5)], color = colors[1], marker = :diamond, markersize = 25, strokewidth = 1.2),
            MarkerElement(points = [Point2f(1.1, 0.5)], color = colors[2], marker = :diamond, markersize = 25, strokewidth = 1.2)]
            #[LineElement(color = :black, linestyle = nothing, points = Point2f[(0.5, 0), (0.5, 1)], linewidth = 2),
    el2 =  [MarkerElement(points = [Point2f(-0.2, 1.1)], color = colors[1], marker = :star6, markersize = 25, strokewidth = 1.2),
                MarkerElement(points = [Point2f(1.1, 1.1)], color = colors[2], marker = :star6, markersize = 25, strokewidth = 1.2)]
            
    el3 = [MarkerElement(points = [Point2f(-0.1, 0.5)], color = colors[1], marker = :diamond, markersize = 25, strokewidth = 1),
            MarkerElement(points = [Point2f(1.2, 0.5)], color = colors[1], marker = :star6, markersize = 25, strokewidth = 1)]
            #[LineElement(color = :black, linestyle = nothing, points = Point2f[(0.5, 0), (0.5, 1)], linewidth = 2),
    el4 =  [MarkerElement(points = [Point2f(-0.1, 0.5)], color = colors[2], marker = :diamond, markersize = 25, strokewidth = 1),
                MarkerElement(points = [Point2f(1.2, 0.5)], color = colors[2], marker = :star6, markersize = 25, strokewidth = 1)]
            #[LineElement(color = :black, linestyle = nothing, points = Point2f[(0.5, 0), (0.5, 1)], linewidth = 2),
    eh = [LineElement(color = :black, linestyle = :dash, linewidth = 4, points = Point2f[(-0.5, 0.5), (1.0, 0.5)])]

    axislegend(ax, [el3,el4,eh],
        ["AvgInst $(cov_temps[1]) K", "AvgInst $(cov_temps[2]) K", "Dulong-Petit"],
        position = :lb, labelsize = 35,
        orientation = :horizontal, framevisible = false, nbanks = 3, patchlabelgap = 20, padding = (20.0f0, 20.0f0, 20.0f0, 20.0f0))

    axislegend(ax, [el1, el2],
        [L"\text{var}(U_n)", L"\sum_{m, m \neq n}\text{cov}(U_n,U_m)"], "Component:",
        position = :rb, labelsize = 30, titlesize = 30,
        orientation = :horizontal, framevisible = false, nbanks = 2, labelhalign = :center, rowgap = 7)
end

size_in_inches = (3, 6.75)
dpi = 300
size_in_pixels = size_in_inches .* dpi
f = Figure(resolution = size_in_pixels);

if pot == "LJ"
    xlims = [0, 2.8]
    dos_max = 0.15
    xticks = [0,0.5,1.0,1.5,2.0,2.5]
    dos_yticks = [0.0, 0.04, 0.08, 0.12]
    mode_ylims = [0.36, 0.6]
    mode_yticks = [0.4,0.45,0.5,0.55]
    cov_ylim = [-0.7,0.80]
    cov_yticks = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75]
elseif pot == "SW"
    xlims = [0, 18.5]
    dos_max = 0.065
    xticks = [0,4,8,12,16]
    dos_yticks = [0.0, 0.02, 0.04, 0.06]
    mode_ylims = [0.47, 0.75]
    mode_yticks = [0.5,0.55,0.6,0.65,0.7]
    cov_ylim = [-0.3, 0.6]
    cov_yticks = [-0.2, 0.0, 0.2, 0.4, 0.5]
end

dos_axis = Axis(f[1,1], limits = (xlims..., 0, dos_max),
                    yticks = dos_yticks, xticklabelsvisible = false, xticks = xticks,
                    ylabel = L"\text{Density of States [1/THz]}", ylabelsize = 40, xlabelsize = 40, 
                    yticklabelsize = 30, xticklabelsize = 30, xgridvisible = false, ygridvisible = false, 
                    xticksmirrored = true, yticksmirrored = true, 
                    xticklabelpad = 4, xtickalign=1, ytickalign = 1)

mode_cv_axis = Axis(f[2,1], limits = (xlims..., mode_ylims...), ylabelsize = 40, xlabelsize = 40,
                    ylabel =  ylabel = L"C_{V,n} / k_{\text{B}}", xticklabelsvisible = false, xticks = xticks,
                    yticks = mode_yticks, yticklabelsize = 30, xticklabelsize = 30, xgridvisible = false, ygridvisible = false,
                    xticksmirrored = true, yticksmirrored = true, xticklabelpad = 4, xtickalign=1, ytickalign = 1)

cov_axis = Axis(f[3,1], xlabel = L"$\omega_n / (2\pi)$ [THz]", ylabel =  ylabel = L"C_{V,n} / k_{\text{B}} \text{ Component}",
                    ylabelsize = 40, xlabelsize = 40, yticklabelsize = 30, xticklabelsize = 30,
                    xgridvisible = false, ygridvisible = false, xticks = xticks, yticks = cov_yticks,
                    xticksmirrored = true, yticksmirrored = true, xtickwidth = 1.5, ytickwidth = 1.5,
                    xticklabelpad = 4, xtickalign=1, ytickalign = 1, limits = (xlims..., cov_ylim...))



    
linkxaxes!(dos_axis, mode_cv_axis)
linkxaxes!(mode_cv_axis, cov_axis)

xspace = maximum(tight_xticklabel_spacing!, [dos_axis, mode_cv_axis, cov_axis])

dos_axis.xticklabelspace = xspace
mode_cv_axis.xticklabelspace = xspace
cov_axis.xticklabelspace = xspace

DOS_PLOT(dos_axis, pot, dos_temp)
mode_cv_plot(mode_cv_axis, pot, xlims[2], mode_temps)
cov_plot(cov_axis, pot, xlims[2])

save(joinpath(outpath, "$(pot)_VERTICAL.svg"), f)
save(joinpath(outpath, "$(pot)_VERTICAL.png"), f)
save(joinpath(outpath, "$(pot)_VERTICAL.pdf"), f)
