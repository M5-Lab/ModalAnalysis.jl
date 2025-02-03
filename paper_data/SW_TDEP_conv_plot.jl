using JLD2
using CairoMakie

temps = [100,1600]
colors = Dict(100 => :blue, 1600 => :red)
# n_samps = [10,20,30,40,50,75,100,150,200,300,400]
n_samps = [100,150,200,300,400]


size_in_inches = (3, 2.25)
dpi = 300
size_in_pixels = size_in_inches .* dpi

f = Figure(resolution = size_in_pixels);
ax = Axis(f[1,1], xlabel = "Number of Samples", ylabel = L"C_{V,U} / (Nk_{\text{B}})",
    ylabelsize = 40, xlabelsize = 40, yticklabelsize = 30, xticklabelsize = 30,
    yticks = [0.45,0.5,0.55,0.6], xticks = [0,100,200,300,400],
    xgridvisible = false, ygridvisible = false, xticksmirrored = true, yticksmirrored = true, 
    xticklabelpad = 4, xtickalign=1, ytickalign = 1)

plts = []
for temp in temps
    base_path = raw"\\mcgaugheynas.lan.local.cmu.edu\mcgaughey-lab\emeitz\Data\IFC_Convergence"
    base_path *= "/SW_3UC_TDEP_HeatCapConv_$(temp)"

    MD_cv_arr = zeros(length(n_samps)); 
    MD_se_arr = zeros(length(n_samps));
    TEP_cv_arr = zeros(length(n_samps)); 
    TEP_se_arr = zeros(length(n_samps));

    for (i,N) in enumerate(n_samps)

        data_path = joinpath(base_path, "T$(temp)_N$(N)", "cv_data_averaged.jld2")
        MD_cv, TEP_cv, MD_se, TEP_se = load(data_path, "cv_MD_total_avg", "TEP_cv_total_avg", "MD_cv_total_std_err", "TEP_cv_total_std_err")

        MD_cv_arr[i] = MD_cv
        MD_se_arr[i] = MD_se
        TEP_cv_arr[i] = TEP_cv
        TEP_se_arr[i] = TEP_se
    end  
    
    errorbars!(n_samps, TEP_cv_arr, TEP_se_arr, TEP_se_arr, whiskerwidth = 10, direction = :y)
    push!(plts, scatter!(n_samps, TEP_cv_arr, markersize = 40, color = colors[temp], strokecolor = :black, strokewidth = 2));
end

ylims!(0.45,0.62)
xlims!(0,410)
# s1 = lines!(range(0,400), MD_cv_arr[1]*ones(401), markersize = 40, linewidth = 6);
# s2 = scatter!(n_samps, TEP_cv_arr, markersize = 40, color = :red);

# errorbars!(n_samps, MD_cv_arr, MD_se_arr, MD_se_arr, whiskerwidth = 10, direction = :y)
# errorbars!(n_samps, TEP_cv_arr, TEP_se_arr, TEP_se_arr, whiskerwidth = 10, direction = :y)
base_path = raw"\\mcgaugheynas.lan.local.cmu.edu\mcgaughey-lab\emeitz\Data\IFC_Convergence"

axislegend(ax, plts, ["TDEP-SS 100 K", "TDEP-SS 1600 K"], position = :rt, labelsize = 35, orientation = :vertical, framevisible = false)
save(joinpath(base_path,"SW_TDEP_IFC_Conv.png"), f)




