export INMA


function INMA(ld::LammpsDump, pair_potential::Potential, potential_eng_MD, masses, max_deviation, out_basepath, T_des, kB)
   
    #Assumes 3D
    box_sizes = [ld.header_data["L_x"][2],ld.header_data["L_y"][2],ld.header_data["L_z"][2]]
    N_modes = 3*length(masses)

    recalculate_INMs = true
    recalculation_idxs = [1]

    #Pre-allocate
    f0_nmc = zeros(N_modes)
    mode_potential_order2 = zeros(N_modes, N_steps)
    mode_potential_order3 = zeros(N_modes, N_steps)
    total_eng_INM = similar(potential_eng_MD)

    recalc_counter = 1
    for i in eachindex(potential_eng_MD)

        ld = parse_timestep!(ld, i)

        if recalculate_INMs
            #Build system with current positions
            sys = SuperCellSystem(ld.data_storage, masses, box_sizes, "x", "y", "z")

            # Initialize INMs to 3rd Order
            freqs_sq, phi, K3 = get_modal_data(sys, pair_potential)
            
            #Set new reference positions to calculate displacements 
            reference_data = copy(ld.data_storage)
            f0_nmc .= reduce(vcat, eachrow(Matrix(reference_data[!,["fx","fy","fz"]]))./(sqrt.(masses)) )
            f0_nmc = (phi') * f0_nmc

            recalculate_INMs = false
            reference_idx = i

            #Create new group
            group_name = "INM Set $(recalc_counter)"
            jldopen(joinpath(out_basepath, "INMA.jld2"), "a+") do file
                mygroup = JLD2.Group(file, group_name)
                mygroup["freqs_sq"] = freqs_sq
                mygroup["phi"] = phi
            end

        end

        
        #Calculate displacements
        ld = unwrap_coordinates!(ld, reference_data, box_sizes)
        disp = ld.data_storage[!, ["x","y","z"]] .- reference_data[!, ["x","y","z"]]
        for i in eachindex(masses) disp[i] .*= sqrt(masses[i]) end
        disp_mw = reduce(vcat, disp_mw)

        #Convert displacements to mode amplitudes.
        q_anharmonic = phi' * disp_mw;

        #Calculate energy from INMs at timestep i
        mode_potential_order2 .= (-f0_nmc.*q_anharmonic) .+ 0.5.*(freqs_sq .* (q_anharmonic.^2))
        mode_potential_order3 .= mode_potential_order2 .+ U_TEP3_n.(Ref(K3), Ref(q_anharmonic), range(1,N_modes))
        INM_energy = sum(mode_potential_order3) + potential_eng_MD[reference_idx]
        
        dist_btwn = abs(potential_eng_MD[i] - INM_energy)
        total_eng_INM[i] = INM_energy
        
        #Append energies to file
        jldopen(joinpath(out_basepath, "INMA.jld2"), "a+") do file
            file[group_name*"/Step $i/mode_potential_order2"] = mode_potential_order2
            file[group_name*"/Step $i/mode_potential_order3"] = mode_potential_order3
        end
            
        if dist_btwn > max_deviation
            recalculate_INMs = true
            push!(recalculation_idxs, i + 1)
            println(i)
            recalc_counter += 1
        end

    end

    steps_btwn_reset = [recalculation_idxs[i+1] - recalculation_idxs[i] for i in range(1,length(recalculation_idxs)-1)]

    jldopen(joinpath(out_basepath, "INMA.jld2"), "a+") do file
        file["total_eng_INM"] = total_eng_INM
        file["steps_btwn_reset"] = steps_btwn_reset
        file["avg_reset_time"] = mean(steps_btwn_reset)
        file["potential_eng_MD"] = potential_eng_MD
        file["cv_MD_total"] = var(potential_eng_MD)/(kB*T_des*T_des)
        file["cv_TEP_total"] = var(total_eng_INM)/(kB*T_des*T_des)
        file["params/max_deviation"] = max_deviation
        file["params/T_des"] = ustrip(T_des)
        file["params/kB"] = ustrip(kB)
    end

    return total_eng_INM, steps_btwn_reset

end
