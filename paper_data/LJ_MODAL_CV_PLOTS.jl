using JLD2
using CairoMakie
using Statistics

calcs = ["AVGIFC_CLEAN","ALM_StepByStep", "OneShot", "0K"]
paths = [joinpath(@__DIR__,"DATA/LJ_NMA/Modal"),
        joinpath(@__DIR__,"DATA/LJ_NMA/Bulk"),
        joinpath(@__DIR__,"DATA/LJ_NMA/Bulk"),
        joinpath(@__DIR__,"DATA/LJ_NMA/Bulk")] #for AvgINM use Modal folder, for others use Bulk
base_outpath = joinpath(@__DIR__,"LJ_PLOTS")

temps = [10,40,80]
colors = ["#a3f55b","#1dd6f2", "#f21b6a"]
legend_labels = ["10 K", "40 K", "80 K"]
y_maxs = [0.56, 0.52, 0.58, 0.52]
y_mins = [0.36, 0.18, 0.36, 0.15]
yticks = [[0.4, 0.45,0.5,0.55],[0.25, 0.3,0.35,0.4,0.45,0.5],[0.4, 0.45,0.5,0.55],[0.2,0.3,0.4,0.5]]


for (c,calc) in enumerate(calcs)
    plts = []

    size_in_inches = (3, 2.25)
    dpi = 300
    size_in_pixels = size_in_inches .* dpi

    filenames = ["LJ_$(calc)_mode_cv_10.svg", "LJ_$(calc)_mode_cv_1040.svg","LJ_$(calc)_mode_cv_104080.svg"]


    f = Figure(resolution = size_in_pixels);
    ax = Axis(f[1,1], xlabel = L"$\omega_n / (2\pi)$ [THz]", ylabel = L"C_{V,n} / k_{\text{B}}",
        ylabelsize = 40, xlabelsize = 40, yticklabelsize = 30, xticklabelsize = 30,
        yticks = yticks[c], xticks = [0,0.5,1.0,1.5,2.0,2.5],
        xgridvisible = false, ygridvisible = false, xticksmirrored = true, yticksmirrored = true, 
        xticklabelpad = 4, xtickalign=1, ytickalign = 1)

    xlims!(0,2.8)
    ylims!(y_mins[c],y_maxs[c])

    h = hlines!(0.5, 0, 2.8, color = :black, linestyle = :dash, linewidth = 4)
    eh = [LineElement(color = :black, linestyle = :dash, linewidth = 4, points = Point2f[(-0.5, 0.5), (1.0, 0.5)])]

    axislegend(ax, [eh], ["Dulong-Petit"], 
                position = :rb, labelsize = 35, orientation = :horizontal, framevisible = false)

    for (i,temp) in enumerate(temps)

        # BIN HEAT CAPACITY BY FREQUENCY (ROUND TO 2 DECIMALS)
        freqs_path = joinpath(paths[c], calc, "cv_data_averaged_$(temp)K.jld2")
        freqs_unique, cv_data_avg, cv_se = load(freqs_path, "unique_freqs", "TEP_cv_per_mode_avg", "TEP_cv_per_mode_std_err")

        errorbars!(sqrt(418.4)*freqs_unique./(2*pi), cv_data_avg, cv_se, cv_se, 
            whiskerwidth = 10, direction = :y, color = :black)#, color = colors[i])
        push!(plts, scatter!(sqrt(418.4)*freqs_unique./(2*pi), cv_data_avg, strokewidth = 1, color = colors[i], markersize = 20));

        leg = axislegend(ax, plts, legend_labels[1:i], position = :lb, labelsize = 35, bgcolor  = :white, framevisible = false)
        save(joinpath(base_outpath,filenames[i]), f)
        delete!(leg)

    end
end