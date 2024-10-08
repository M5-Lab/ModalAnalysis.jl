export NM_postprocess, NMA_avg_seeds, make_plots

function NM_postprocess(nma::NormalModeAnalysis, kB; nthreads::Integer = Threads.nthreads(),
     average_identical_freqs = true, freq_digits = 7)

    T = nma.temperature
    energies_path = joinpath(nma.simulation_folder, "ModeEnergies.jld2")
    tep_path = joinpath(nma.simulation_folder, "TEP.jld2")

    potential_eng_MD, potential_eng_TEP, mode_potential_energy =
        load(energies_path, "pot_eng_MD", "total_eng_NM", "mode_potential_energy")
    freqs_sq = load(tep_path, "freqs_sq")
 
    N_modes = length(freqs_sq)

    if any(freqs_sq .< 0.0)
        freqs = imag_to_neg(sqrt.(Complex.(freqs_sq)))
    else
        freqs = sqrt.(freqs_sq)
    end
    freqs = round.(freqs, digits = freq_digits)

    #Bulk heat capacities
    cv_total_MD = var(potential_eng_MD)/(kB*T*T)
    cv_TEP_total = var(potential_eng_TEP)/(kB*T*T)

    cv3_cov, cv3_per_mode, cv3_total = 
        build_cov_matrix(mode_potential_energy, N_modes, nthreads, kB, T)


    if average_identical_freqs
        unique_freqs = unique(freqs)
        cv3_avg_freq = zeros(length(unique_freqs))
        for (i, f) in enumerate(unique_freqs)
            idxs = findall(x -> x == f, freqs)
            cv3_avg_freq[i] = sum(cv3_per_mode[idxs])/length(idxs)
        end
    end

    #Sanity check
    if !isapprox(cv3_total, cv_TEP_total, atol = 1e-4)
        @warn "Sum of modal heat capacities ($(cv3_total)) does not match heat capacity from total TEP energy ($(cv_TEP_total))"
    end

    # run_ks_tests && run_ks_experiments(nma.simulation_folder, N_modes)

    #Save heat capacity data
    if average_identical_freqs
        jldsave(joinpath(nma.simulation_folder, "cv_data.jld2"); 
            cv_total_MD = cv_total_MD, cv_total_MD_norm = cv_total_MD/(N_modes*kB),
            cv3_total = cv3_total, cv3_total_norm = cv3_total/(N_modes*kB),
            cv3_per_mode = cv3_per_mode, cv3_per_mode_norm = cv3_per_mode./kB,
            cv3_avg_freq = cv3_avg_freq, cv3_avg_freq_norm = cv3_avg_freq./kB,
            cv3_cov = cv3_cov, freqs = freqs, cv_TEP_full_dist = cv_TEP_total)
    else
        jldsave(joinpath(nma.simulation_folder, "cv_data.jld2");
        cv_total_MD = cv_total_MD, cv_total_MD_norm = cv_total_MD/(N_modes*kB),
        cv3_total = cv3_total, cv3_total_norm = cv3_total/(N_modes*kB),
        cv3_per_mode = cv3_per_mode, cv3_per_mode_norm = cv3_per_mode./kB,
        cv3_cov = cv3_cov, freqs = freqs, cv_TEP_full_dist = cv_TEP_total)
    end


end

"""
Expects `per_mode_potential` to be (N_modes x N_samples)
"""
function build_cov_matrix(per_mode_potential::Matrix, N_modes::Integer, nthreads::Integer, kB, T)

    cv3_cov = zeros(N_modes, N_modes)
    per_mode_potential = permutedims(per_mode_potential) #flip to get column major access
    Threads.@threads for thread_id in 1:nthreads
        for n in thread_id:nthreads:N_modes

            @views cv3_cov[n,n] = var(per_mode_potential[:,n])/(kB*T*T)

            #Covariance terms
            for m in range(n+1,N_modes)
                @views cv3_cov[n,m] = cov(per_mode_potential[:,n], per_mode_potential[:,m])/(kB*T*T)
                cv3_cov[m,n] = cv3_cov[n,m]
            end
        end
    end

    cv3_per_mode = sum(cv3_cov, dims = 2);
    cv3_total = sum(cv3_per_mode)

    return cv3_cov, cv3_per_mode, cv3_total
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

"Goes through multiple seeds of NMA data and averages into one dataset. 
Looks for the cv_data.jld2 file generated by NM analysis at: \"/<basepath>/<seed_subfolder><N>\". 
Where `basepath` and `seed_subfolder` are function parameters"
function NMA_avg_seeds(basepath, n_seeds, T; 
    seed_subfolder::String = "seed", seeds_zero_indexed::Bool = true)

    freqs = load(joinpath(basepath, "$(seed_subfolder)$(1-seeds_zero_indexed)/cv_data.jld2"), "freqs")
    unique_freqs = unique(freqs)

    MD_cv_total = []
    TEP_cv_total = []
    cv3_by_freq = [Float64[] for _ in 1:length(unique_freqs)]

    for seed in 1:n_seeds
        seed_path = joinpath(basepath,"$(seed_subfolder)$(seed-seeds_zero_indexed)/cv_data.jld2")
        cv_total_MD_norm, cv3_total_norm, cv3_per_mode_norm =
            load(seed_path, "cv_total_MD_norm", "cv3_total_norm", "cv3_per_mode_norm")

        push!(MD_cv_total, cv_total_MD_norm)
        push!(TEP_cv_total, cv3_total_norm)
        
        #Sort heat capacities by frequency (will do nothing if freqs is all unique)
        for (i, f) in enumerate(unique_freqs)
            idxs = findall(x -> x == f, freqs)
            append!(cv3_by_freq[i], cv3_per_mode_norm[idxs])
        end
    end

    MD_cv_total_avg = mean(MD_cv_total)
    MD_cv_total_std_err = std(MD_cv_total)/sqrt(n_seeds)
    TEP_cv_total_avg = mean(TEP_cv_total)
    TEP_cv_total_std_err = std(TEP_cv_total)/sqrt(n_seeds)
    #Average heat capacities at each freq
    TEP_cv_per_mode_avg =  mean.(cv3_by_freq)
    #StdErr of heat capacity at each freq
    TEP_cv_per_mode_std_err = std.(cv3_by_freq)./sqrt(n_seeds) #./sqrt.(length.(cv3_by_freq))
    samples_per_freq = length.(cv3_by_freq)

    f = Figure()
    ax = Axis(f[1,1], xlabel = "Mode Frequency", ylabel = L"\text{Mode Heat Capacity / }k_{\text{B}}",
        title = "$(T)K", titlesize = 20, yticks = [0.0,0.25,0.5], ylabelsize = 30, xlabelsize = 30,
        yticklabelsize = 20, xticklabelsize = 20)
    (n_seeds > 1) && errorbars!(unique_freqs, TEP_cv_per_mode_avg, TEP_cv_per_mode_std_err, TEP_cv_per_mode_std_err, whiskerwidth = 3, direction = :y)
    scatter!(unique_freqs, TEP_cv_per_mode_avg);
    save(joinpath(basepath, "freqs_$(T)K.svg"), f)

    jldsave(joinpath(basepath, "cv_data_averaged.jld2"); cv_MD_total_avg = MD_cv_total_avg,
        MD_cv_total_std_err = MD_cv_total_std_err, TEP_cv_total_avg = TEP_cv_total_avg, TEP_cv_total_std_err = TEP_cv_total_std_err,
        TEP_cv_per_mode_avg = TEP_cv_per_mode_avg, TEP_cv_per_mode_std_err = TEP_cv_per_mode_std_err,
        unique_freqs = unique_freqs, freqs_all = freqs, samples_per_freq = samples_per_freq)
    
end


# Makes plots that cannot be made in analysis of MD data due to the non-thread-safe nature of plots in Julia
function make_plots(simulation_folder::String, T; normalize_cv = true, average_identical_freqs = true)

    mode_energies_path = joinpath(simulation_folder, "ModeEnergies.jld2")
    cv_data_path = joinpath(simulation_folder, "cv_data.jld2")

    pot_eng_MD, potential_eng_TEP =
        load(mode_energies_path, "pot_eng_MD", "total_eng_NM")

    system_energy_hist(simulation_folder, pot_eng_MD, potential_eng_TEP)

    if average_identical_freqs
        freqs_all = load(cv_data_path, "freqs")
        freqs_unique = unique(freqs_all)
        if normalize_cv
            cv3_avg_freq_norm, cv3_per_mode_norm = load(cv_data_path, "cv3_avg_freq_norm","cv3_per_mode_norm")
            heat_cap_per_mode_scatter(simulation_folder, freqs_all, cv3_per_mode_norm, T, "heat_cap_per_mode_norm")
            heat_cap_per_mode_scatter(simulation_folder, freqs_unique, cv3_avg_freq_norm, T, "heat_cap_per_mode_avg_freq_norm")
        else
            cv3_avg_freq, cv3_per_mode = load(cv_data_path, "cv3_avg_freq","cv3_per_mode")
            heat_cap_per_mode_scatter(simulation_folder, freqs_all, cv3_per_mode, T, "heat_cap_per_mode")
            heat_cap_per_mode_scatter(simulation_folder, freqs_unique, cv3_avg_freq, T, "heat_cap_per_mode_avg_freq")
        end
    else
        freqs_all = load(cv_data_path, "freqs")
        if normalize_cv
            cv3_per_mode_norm = load(cv_data_path, "cv3_per_mode_norm")
            heat_cap_per_mode_scatter(simulation_folder, freqs_all, cv3_per_mode_norm, T, "heat_cap_per_mode_norm")
        else
            cv3_per_mode = load(cv_data_path, "cv3_per_mode")
            heat_cap_per_mode_scatter(simulation_folder, freqs_all, cv3_per_mode, T, "heat_cap_per_mode")
        end
    end


end

function system_energy_hist(simulation_folder::String, pot_eng_MD, potential_eng_TEP)

    #System level energy histograms
    f = Figure()
    ax = Axis(f[1,1], xlabel = "Potential Energy", ylabel = "PDF")
    s1 = stephist!(pot_eng_MD, normalization = :pdf)
    s2 = stephist!(potential_eng_TEP, normalization = :pdf)
    Legend(f[1,2], [s1,s2], ["MD", "TEP"], "Energy Calculator")
    save(joinpath(simulation_folder,"pot_eng_hist.svg"), f)

end

function heat_cap_per_mode_scatter(simulation_folder::String, freqs, cv3_per_mode, T, plot_name::String)

    f = Figure()
    ax = Axis(f[1,1], xlabel = "Mode Frequency", ylabel = L"\text{Mode Heat Capacity / }k_{\text{B}}",
        title = "Heat Capacity per Mode: T = $T")
    scatter!(freqs, vec(cv3_per_mode))
    save(joinpath(simulation_folder,"$(plot_name).svg"), f)

end

# #############################################
# """
#  -α: p-value required to reject null-hypothesis
#  -n_experiments: Number of times to randomly sample a pair of two distributions
#  -n_samples_per_hist: Number of samples taken in each experiment (with replacement)
#  -required_pass_rate: Pair of histograms is deemed the same if this many pairs of KS test fail to reject H₀
# """
# function run_ks_experiments(simulation_folder::String, N_modes::Integer;
#      α = 0.05, n_experiments = 20, required_pass_rate = 0.5, n_samples_per_hist = 10000)
#     energy_path = joinpath(simulation_folder, "ModeEnergies.jld2")

#     energies_order3 = load(energy_path, "mode_potential_order3");

#     p_val_matrix = zeros(N_modes, N_modes)
#     D_matrix = zeros(N_modes, N_modes)
#     pass_matrix = zeros(N_modes, N_modes)

#     Threads.@threads for i in range(1, N_modes)
#         sample1_storage = zeros(n_samples_per_hist)
#         sample2_storage = zeros(n_samples_per_hist)
#         for j in range(i, N_modes)
#             isSameDist, avg_p_val, avg_δ = @views ks_experiment(energies_order3[i,:], energies_order3[j,:], 
#                 sample1_storage, sample2_storage, n_experiments, required_pass_rate, α)
#             p_val_matrix[i,j] = avg_p_val
#             p_val_matrix[j,i] = p_val_matrix[i,j]
#             D_matrix[i,j] = avg_δ
#             D_matrix[j,i] = avg_δ
#             pass_matrix[i,j] = isSameDist
#             pass_matrix[j,i] = isSameDist
#         end
#     end

#     save(joinpath(simulation_folder, "ks_p_val_matrix.png"), colorview(Gray, p_val_matrix))
#     save(joinpath(simulation_folder, "ks_pass_matrix.png"), colorview(Gray, pass_matrix))
#     save(joinpath(simulation_folder, "ks_dist_matrix.png"), colorview(Gray, D_matrix))

# end



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


# function resample_CLT(energy)
#     N_steps = length(energy)
#     N_samples = N_steps ÷ 10
#     M = 500
#     energy_samples = sample(energy,(N_samples,M))
#     energy_samples = mean(energy_samples,dims = 2)

#     return M, energy_samples
# end