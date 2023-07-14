export NM_postprocess

function NM_postprocess(energies_path::String, tep_path::String, 
        kB, T, N_modes; nthreads::Integer = Threads.nthreads(), average_identical_freqs = true)

    NMA_filepath = joinpath(energies_path, "ModeEnergies.jld2")
    TEP_filepath = joinpath(tep_path, "TEP.jld2")

    potential_eng_MD, potential_eng_TEP, mode_potential_order3 =
        load(NMA_filepath, "potential_eng_MD", "total_eng_NM", "mode_potential_order3")
    freqs_sq, phi = load(TEP_filepath, "freqs_sq", "phi")

    if any(freqs_sq .< 0.0)
        freqs = imag_to_neg(sqrt.(Complex.(freqs_sq)))
    else
        freqs = sqrt.(freqs_sq)
    end

    #Bulk heat capacities
    cv_total_MD = var(potential_eng_MD)/(kB*T*T)
    cv_TEP_total = var(potential_eng_TEP)/(kB*T*T)

    #Save system level energy histograms
    f = Figure()
    Axis(f[1,1], xlabel = "Potential Energy", ylabel = "Count")
    s1 = stephist!(potential_eng_MD); s2 = stephist!(potential_eng_TEP)
    Legend(f[1,2], [s1,s2], ["MD", "TEP"], "Energy Calculator")
    save(joinpath(energies_path,"pot_eng_hist.svg"), f)

    #Write per-mode data to file
    cv3_cov = zeros(N_modes, N_modes)
    Threads.@threads for thread_id in 1:nthreads
        for n in thread_id:nthreads:N_modes

            cv3_cov[n,n] = var(mode_potential_order3[n,:])/(kB*T*T)

            #Covariance terms
            for m in range(n+1,N_modes)
                @views cv3_cov[n,m] = cov(mode_potential_order3[n,:], mode_potential_order3[m,:])/(kB*T*T)
                cv3_cov[m,n] = cv3_cov[n,m]
            end
        end
    end

    cv3_per_mode= sum(cv3_cov, dims = 2);
    cv3_total = sum(cv3_per_mode)

    if average_identical_freqs
        unique_freqs = unique(round.(freqs, sigdigits = 5))
        cv3_avg = zeros(length(unique_freqs))
        for (i, f) in enumerate(unique_freqs)
            idxs = findall(x -> x == f, freqs)
            cv3_avg[i] = sum(cv3_per_mode[idxs])/length(idxs)
        end

        f = Figure()
        Axis(f[1,1], xlabel = "Mode Frequency", ylabel = L"Mode Heat Capacity / k_{\text{B}}",
            title = "Heat Capacity per Mode: T = $T (avg by freq)")
        scatter!(unique_freqs, cv3_avg);
        save(joinpath(energies_path,"heat_cap_per_mode_avg_freq.svg"), f)
    end

    #Plot mode heat capacities -- raw output
    f = Figure()
    Axis(f[1,1], xlabel = "Mode Frequency", ylabel = L"Mode Heat Capacity / k_{\text{B}}",
        title = "Heat Capacity per Mode: T = $T")
    scatter!(freqs,cv3_per_mode);
    save(joinpath(energies_path,"heat_cap_per_mode.svg"), f)
    
    #Sanity check
    if !isapprox(cv3_total, cv_TEP_total, atol = 1e-4)
        @warn "Sum of modal heat capacities ($(cv3_total)) does not match heat capacity from total TEP energy ($(cv_TEP_total))"
    end

    #Save heat capacity data
    jldsave(joinpath(NM_analysis_folder, "cv_data.jld2"), 
        cv_total_MD = cv_total_MD, cv3_total = cv3_total,
        cv3_per_mode = cv3_per_mode, cv3_cov = cv3_cov)

end

function average_seeds()

end

function imag_to_neg(freqs)
    freqs_float = zeros(size(freqs))
    for i in eachindex(freqs)
        if real(freqs[i]) == 0
            freqs_float[i] = -1*imag(freqs[i])
        else
            freqs_float[i] = real(freqs[i])
        end
    end
    return freqs_float
end

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


function resample_CLT(energy)
    N_steps = length(energy)
    N_samples = N_steps ÷ 10
    M = 500
    energy_samples = sample(energy,(N_samples,M))
    energy_samples = mean(energy_samples,dims = 2)

    return M, energy_samples
end