# using HypothesisTests
# using JLD2
# using Plots
# using ImageView
# using StatsBase

# base_path = raw"C:\Users\ejmei\Desktop\INM_test\MD_simulations_heroic\sim_T_0.1_LJ"
# n_seeds = length(readdir(base_path))

# @info "Num Seeds: $(n_seeds)"

# #Look at MD data
# pe_all = []
# for i in range(1, n_seeds)
#     pe = load(joinpath(base_path,"seed$i","MD_energies.jld2"), "pe")
#     push!(pe_all, pe)
# end
# pe_all = Iterators.flatten(pe_all);
# # histogram(collect(pe_all), dpi = 300)

# #Average Heat Capacity Data
# N_modes = 48
# cov_matrix_all = zeros(N_modes, N_modes)
# cv_actual_avg = 0.0
# cv3_total_avg = 0.0
# for i in range(1, n_seeds)
#     cv3_cov, cv3_total, cv_total_actual = load(joinpath(base_path,"seed$i","NM_Analysis","cv_data.jld2"),
#          "cv3_cov", "cv3_total", "cv_total_actual")
#     cov_matrix_all += cv3_cov
#     cv3_total_avg += cv3_total
#     cv_actual_avg += cv_total_actual
# end
# cov_matrix_all ./= n_seeds #Avg cv per mode from TEP3
# cv_actual_avg /= n_seeds #Avg cv from MD energy
# cv3_total_avg /= n_seeds #Avg cv from TEP3 energy

# cv3 = sum(cov_matrix_all, dims = 2)

# #Color code by e-vec direction
# freqs, phi = load(joinpath(base_path,"seed1","NM_Analysis","NormalModes.jld2"), "freqs", "phi")
# freqs = real(freqs)

# scatter(freqs, cv3)
# xlabel!("Mode Frequency")
# ylabel!("Mode Heat Capacity")

# #Average on Duplicate Freqs

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
# p_val_matrix = zeros(n_modes, n_modes)
# D_matrix = zeros(n_modes, n_modes)
# pass_matrix = zeros(n_modes, n_modes)

# # Number of times to randomly sample a pair of two distributions
# n_experiments = 20
# # Number of samples taken in each experiment (with replacement)
# n_samples_per_hist = 10_000
# # Pair of histograms is deemed the same if this many pairs KS test fail to reject H₀
# required_pass_rate = 0.5
# # P-value needed to reject null-hypothesis
# α = 0.05

# Threads.@threads for i in range(1, n_modes)
#     sample1_storage = zeros(n_samples_per_hist)
#     sample2_storage = zeros(n_samples_per_hist)
#     for j in range(i, n_modes)
#         isSameDist, avg_p_val, avg_δ = @views ks_experiment(energies_order3[i,:], energies_order3[j,:], 
#             sample1_storage, sample2_storage, n_experiments, required_pass_rate, α)
#         p_val_matrix[i,j] = avg_p_val
#         p_val_matrix[j,i] = p_val_matrix[i,j]
#         D_matrix[i,j] = avg_δ
#         D_matrix[j,i] = avg_δ
#         pass_matrix[i,j] = isSameDist
#         pass_matrix[j,i] = isSameDist
#     end
# end

# imshow(p_val_matrix)
# imshow(pass_matrix)
# imshow(D_matrix)


# function ks_experiment(data1, data2, sample1_storage, sample2_storage, n_experiments, required_pass_rate, α)
#     n_same_dist = 0
#     p_vals = zeros(n_experiments)
#     δs = zeros(n_experiments)
#     for i in 1:n_experiments
#         sample1_storage = sample!(data1, sample1_storage)
#         sample2_storage = sample!(data2, sample2_storage)
#         ks_res = ApproximateTwoSampleKSTest(sample1_storage, sample2_storage)
#         p_value = pvalue(ks_res)
#         p_vals[i] = p_value
#         δs[i] = ks_res.δ
#         if p_value > α #fail to reject null --> same distributions
#             n_same_dist += 1
#         end
#     end

#     return ((n_same_dist/n_experiments) > required_pass_rate), sum(p_vals)/n_experiments, sum(δs)/n_experiments
# end
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