# using HypothesisTests
# using JLD2
# using Plots
# using Images, ImageView

# base_path = raw"C:\Users\ejmei\Desktop\INM_test\MD_simulations_heroic\sim_T_0.1_LJ"
# n_seeds = length(readdir(base_path))

# @info "Num Seeds: $(n_seeds)"


# #Average Heat Capacity Data
# N_modes = 48
# cov_matrix_all = zeros(N_modes, N_modes)

# for i in range(1, n_seeds)
#     cov_matrix_all += load(joinpath(base_path,"seed$i","NM_Analysis","cv_data.jld2"), "cv3_cov")
# end
# cov_matrix_all ./= n_seeds

# cv3 = sum(cov_matrix_all, dims = 2)

# #For NMA all have same modes
# freqs = load(joinpath(base_path,"seed1","NM_Analysis","NormalModes.jld2"), "freqs")
# freqs = real(freqs)

# freqs_unique = unique(round.(freqs,sigdigits = 5))
# cv_avg = (cv3[2:2:end-1] .+ cv3[3:2:end-1]) ./ 2
# prepend!(cv_avg, cv3[1])
# push!(cv_avg, cv3[end])

# plt = scatter(freqs_unique, cv_avg, dpi = 300, ylim = (0.0,0.55))
# xlabel!("Mode Frequency")
# ylabel!("Mode Heat Capacity")
# title!("100 Seeds")
# display(plt)

# imshow(cov_matrix_all)
# #############################################

# #KS TEST -- Just do on one seed
# energy_path = joinpath(base_path, "seed1", "NM_Analysis", "NM_energies.jld2")

# energies_order3 = load(energy_path, "order3");

# n_modes = 48
# ks_matrix = zeros(n_modes, n_modes)
# D_matrix = zeros(n_modes, n_modes)

# Threads.@threads for i in range(1, n_modes)
#     for j in range(i, n_modes)
#         ks_res = ApproximateTwoSampleKSTest(energies_order3[i,:], energies_order3[j,:])
#         ks_matrix[i,j] = @views pvalue(ks_res)
#         ks_matrix[j,i] = ks_matrix[i,j]
#         D_matrix[i,j] = ks_res.δ
#         D_matrix[j,i] = ks_res.δ
#     end
# end

# imshow(ks_matrix)

### Bin energies


# function bin_energies()

#     dω = 0.5
#     bin_centers = [0.5*dω*i for i in range(-15,25)]
#     energy_bins_2nd_order = Vector{Vector{Float64}}(undef,length(bin_centers))
#     energy_bins_3rd_order = Vector{Vector{Float64}}(undef,length(bin_centers))


#     for n in range(1,N_atoms) #technically N_modes
#         ω = freqs[n]
#         bin_idx = argmin(abs.(bin_centers .- ω))
        
#         #TODO: JUST WRITE TO FILE DONT BIN HERE
#         if !isassigned(energy_bins_2nd_order, bin_idx)
#             energy_bins_2nd_order[bin_idx] = [mode_potential_order2[n]]
#             energy_bins_3rd_order[bin_idx] = [mode_potential_order3[n]]
#         else
#             push!(energy_bins_2nd_order[bin_idx], mode_potential_order2[n])
#             push!(energy_bins_3rd_order[bin_idx], mode_potential_order3[n])
#         end
#     end  
# end