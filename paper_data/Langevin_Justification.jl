using DelimitedFiles
using CairoMakie
using StatsBase

outpath = "C:/Users/ejmei/Box/MyPapers/SolidModalHeatCap/Figs"

SW_100K_path = "Z:/emeitz/Data/NMA/SW/SW_3UC_NMA_CLEANAvgIFC_LANGEVIN/T100"
SW_1300K_path = "Z:/emeitz/Data/NMA/SW/SW_3UC_NMA_CLEANAvgIFC_LANGEVIN/T1300"

LJ_10K_path = "Z:/emeitz/Data/NMA/LJ/NMA_LJ_AvgINM/10K"
LJ_80K_path = "Z:/emeitz/Data/NMA/LJ/NMA_LJ_AvgINM/80K"


n_seeds = 50

paths = [LJ_10K_path, LJ_80K_path, SW_100K_path, SW_1300K_path]

percent_err = zeros((length(paths), n_seeds))

for (i,path) in enumerate(paths)
    for seed in 1:n_seeds
        total_path = joinpath(path, "seed$(seed-1)", "thermo_data.txt")
        data = readdlm(total_path, comments = true)
        actual = var(data[:,3] .+ data[:,4])
        approx = var(data[:,3]) + var(data[:,4])
        percent_err[i, seed] = 100 * abs(approx - actual) / actual
    end
end

avg_percent_err = mean(percent_err, dims = 2)

size_in_inches = (3, 2.25)
dpi = 300
size_in_pixels = size_in_inches .* dpi

f,ax,bp = barplot(vec(avg_percent_err),
    strokecolor = :black,
    strokewidth = 1,
    axis = (
        xticks = (1:4, ["LJ 10K", "LJ 80K", "SW 100K", "SW 1300K"]),
        ylabel = "Percent Error",
        limits = (0.5,4.5,0.0,1.1),
        yticks = [0.0, 0.25, 0.5, 0.75, 1.0],
        xgridvisible = false,
        ygridvisible = false,
        ylabelsize = 40, xlabelsize = 40, yticklabelsize = 30, xticklabelsize = 30
    ), 
    figure = (
        resolution = size_in_pixels,
    ),
    bar_width = 0.5, color = [:blue, :red, :green, :purple]
)

save(joinpath(outpath, "VarApproxPlot.png"), f)