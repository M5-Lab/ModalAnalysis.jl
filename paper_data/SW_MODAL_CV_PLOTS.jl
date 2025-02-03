using JLD2
using CairoMakie
using Statistics

calcs = ["0K", "ALM_StepByStep", "OneShot", "AVGIFC_CLEAN"]
paths = [joinpath(@__DIR__,"DATA/SW_NMA/Modal"),
        joinpath(@__DIR__,"DATA/SW_NMA/Bulk"),
        joinpath(@__DIR__,"DATA/SW_NMA/Bulk"),
        joinpath(@__DIR__,"DATA/SW_NMA/Modal")]
base_outpath = joinpath(@__DIR__,"SW_PLOTS")

temps = [100,700,1300]
colors = ["#a3f55b","#1dd6f2", "#f21b6a"]
legend_labels = ["100 K", "700 K", "1300 K"]
y_maxs = [1.15, 0.75, 0.75, 0.75]
yticks = [[0.5, 0.7, 0.9, 1.1],[0.5, 0.55, 0.6, 0.65, 0.7],[0.5, 0.55, 0.6, 0.65, 0.7],[0.5, 0.55, 0.6, 0.65, 0.7]]


for (c,calc) in enumerate(calcs)
    plts = []

    size_in_inches = (3, 2.25)
    dpi = 300
    size_in_pixels = size_in_inches .* dpi

    filenames = ["SW_$(calc)_mode_cv_100.svg", "SW_$(calc)_mode_cv_100700.svg","SW_$(calc)_mode_cv_1007001300.svg"]


    f = Figure(resolution = size_in_pixels);
    ax = Axis(f[1,1], xlabel = L"$\omega_n / (2\pi)$ [THz]", ylabel = L"C_{V,n} / k_{\text{B}}",
        ylabelsize = 40, xlabelsize = 40, yticklabelsize = 30, xticklabelsize = 30,
        yticks = yticks[c], xticks = [0,4,8,12,16],
        xgridvisible = false, ygridvisible = false, xticksmirrored = true, yticksmirrored = true, 
        xticklabelpad = 4, xtickalign=1, ytickalign = 1)

    xlims!(0,18.5)
    ylims!(0.48,y_maxs[c])

    h = hlines!(0.5, 0, 18.5, color = :black, linestyle = :dash, linewidth = 4)
    eh = [LineElement(color = :black, linestyle = :dash, linewidth = 4, points = Point2f[(-0.5, 0.5), (1.0, 0.5)])]

    axislegend(ax, [eh], ["Dulong-Petit"], position = :rt, labelsize = 35, orientation = :horizontal, framevisible = false)

    for (i,temp) in enumerate(temps)

        # BIN HEAT CAPACITY BY FREQUENCY (ROUND TO 2 DECIMALS)
        freqs_path = joinpath(paths[c], calc, "cv_data_averaged_$(temp)K.jld2")
        freqs_unique, cv_data_avg, cv_se = load(freqs_path, "unique_freqs", "TEP_cv_per_mode_avg", "TEP_cv_per_mode_std_err")

        errorbars!(sqrt(9.64897e3)*freqs_unique/(2*pi), cv_data_avg, cv_se, cv_se, 
            whiskerwidth = 10, direction = :y, color = :black)#, color = colors[i])
        push!(plts, scatter!(sqrt(9.64897e3)*freqs_unique./(2*pi), cv_data_avg, strokewidth = 1, color = colors[i], markersize = 18));

        leg = axislegend(ax, plts, legend_labels[1:i], position = :lt, labelsize = 35, backgroundcolor  = :white, framevisible = false)
        save(joinpath(base_outpath,filenames[i]), f)
        delete!(leg)

    end
end