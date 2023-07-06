


# function NM_loop(r_out, r_u_out, pair_potential, potential_eng, masses_all, L, out_filepath)

#     # Initialize NMs to 3rd Order
#     freqs, phi, K3 = calculate_NMS(view(r_out,1,:), pair_potential, masses_all, L)
#     N_modes = length(freqs)
#     N_steps = length(potential_eng)

#     #Pre-allocate
#     mode_potential_order2 = zeros(N_modes,N_steps)
#     mode_potential_order3 = zeros(N_modes,N_steps)
#     total_mode_eng_order2 = zeros(N_steps)
#     total_mode_eng_order3 = zeros(N_steps)

#     #This could be parallelized but each core needs its own set of INMs
#     for i in eachindex(potential_eng)
        
#         #Calculate displacements
#         disp = r_u_out[i,:] .- r_out[1,:]
#         disp_mw = sqrt.(masses_all) .* disp

#         #Convert displacements to mode amplitudes.
#         q_anharmonic = phi' * disp_mw;

#         #Calculate energy from INMs at timestep i
#         mode_potential_order2[:,i] .= 0.5.*((freqs.^2) .* (q_anharmonic.^2))
#         mode_potential_order3[:,i] .= mode_potential_order2[:,i] .+ U_TEP3_n.(Ref(K3), Ref(q_anharmonic), range(1,N_modes))
        
#         @views total_mode_eng_order2[i] = sum(mode_potential_order2[:,i]) + potential_eng[1]
#         @views total_mode_eng_order3[i] = sum(mode_potential_order3[:,i]) + potential_eng[1]
#     end

#     jldsave(joinpath(out_filepath, "NormalModes.jld2"), freqs = freqs, phi = phi)

#     jldsave(joinpath(out_filepath, "NM_energies.jld2"), order2 = mode_potential_order2,
#         order3 = mode_potential_order3, total_order2 = total_mode_eng_order2,
#         total_order3 = total_mode_eng_order3)
    
# end

# function NM_postprocess(NM_analysis_folder, potential_eng_MD, T, N_dof; kB = 1)

#     order2_modal, order3_modal, order2_total, order3_total =
#         load(joinpath(NM_analysis_folder, "NM_energies.jld2"),
#         "order2", "order3", "total_order2", "total_order3")

#     #Calculate order2 and order3 heat capacity
#     cv_total_actual = var(potential_eng_MD)/(kB*T*T)
#     cv2_total = var(order2_total)/(kB*T*T)
#     cv3_total = var(order3_total)/(kB*T*T)

#     #Save system level energy histograms
#     stephist(Any[potential_eng_MD, order2_total, order3_total], line=(1,0.2,:white),
#         fillcolor=[:blue :red :green], fillalpha=0.4, label = ["MD" "TEP2" "TEP3"], dpi = 300)
#     xlabel!("Potential Energy")
#     ylabel!("Count")
#     savefig(joinpath(NM_analysis_folder, "pot_eng_hist.png"))

#     mode_histogram_outpath = joinpath(NM_analysis_folder, "ModeEnergyHistograms")
#     mkpath(mode_histogram_outpath)

#     #Write per-mode data to file
#     cv2_cov = zeros(N_dof, N_dof)
#     cv3_cov = zeros(N_dof, N_dof)
#     for n in range(1,N_dof)

#         cv2_cov[n,n] = var(order2_modal[n,:])/(kB*T*T)
#         cv3_cov[n,n] = var(order3_modal[n,:])/(kB*T*T)

#         @views histogram(Any[order2_modal[n,:], order3_modal[n,:]], line=(1,0.2,:white),
#             fillcolor=[:blue :red], fillalpha=0.4, label = ["Order 2" "Order 3"], dpi = 300)
#         xlabel!("Potential Energy")
#         ylabel!("Count")
#         savefig(joinpath(mode_histogram_outpath, "mode$(n)_hist.png"))

#         #Covariance terms
#         for m in range(n+1,N_dof)
#             @views cv2_cov[n,m] =  cov(order2_modal[n,:], order2_modal[m,:])/(kB*T*T)
#             cv2_cov[m,n] = cv2_cov[n,m]
#             @views cv3_cov[n,m] = cov(order3_modal[n,:], order3_modal[m,:])/(kB*T*T)
#             cv3_cov[m,n] = cv3_cov[n,m]
#         end
#     end

#     jldsave(joinpath(NM_analysis_folder, "cv_data.jld2"), 
#         cv_total_actual = cv_total_actual, cv2_total = cv2_total, cv3_total = cv3_total,
#         cv2_cov = cv2_cov, cv3_cov = cv3_cov)

#     #Return totals
#     return cv_total_actual, cv2_total, cv3_total, sum(cv2_cov), sum(cv3_cov)
# end

