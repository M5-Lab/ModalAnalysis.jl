export INMA


function run(inma::InstantaneousNormalModeAnalysis)

end

function run(inma::InstantaneousNormalModeAnalysis, mcc_block_size::Integer)

end


function INMA(inma::InstantaneousNormalModeAnalysis, max_deviation)
    
    N_modes = length(freqs_sq) #TODO NOT KNOW AT START
    N_atoms = length(inma.atom_masses)

    mass_sqrt = sqrt.(inma.atom_masses)


    #Assumes 3D
    box_sizes = [inma.ld.header_data["L_x"][2],inma.ld.header_data["L_y"][2],inma.ld.header_data["L_z"][2]]

    recalculate_INMs = true
    recalculation_idxs = [1]

    dump_file = open(inma.ld.path, "r")
    posn_cols = [inma.ld.col_idxs["x"],inma.ld.col_idxs["y"],inma.ld.col_idxs["z"]]

    #Pre-allocate
    f0_nmc = zeros(N_modes)
    reference_posns = zeros(N_atoms, 3)
    current_positions = zeros(N_atoms, 3)
    mode_potential_order3 = zeros(N_modes, inma.ld.n_samples)
    total_eng_INM = zeros(inma.ld.n_samples)

    recalc_counter = 1
    for i in 1:inma.ld.n_samples

        parse_next_timestep!(current_positions, inma.ld, dump_file, posn_cols)

        if recalculate_INMs
            # Initialize INMs to 3rd Order
            #TODO BREAK OUT VERSION WITH MCC_BLOCK_SIZE, write version that keeps K3 on GPU
            freqs_sq, phi, K3 = get_modal_data(inma.sys, inma.pot)
            
            #Set new reference positions to calculate displacements 
            reference_posns .= Matrix(nma.ld.data_storage[!, ["x","y","z"]])
            f0_nmc .= reduce(vcat, eachrow(Matrix(nma.ld.data_storage[!,["fx","fy","fz"]]))./(sqrt.(masses)) ) #TODO replace sqrt_mass
            f0_nmc = (phi') * f0_nmc

            recalculate_INMs = false
            reference_idx = i

            #Create new group
            group_name = "INM$(recalc_counter)"
            jldopen(joinpath(out_basepath, "INMA.jld2"), "a+") do file
                mygroup = JLD2.Group(file, group_name)
                mygroup["freqs_sq"] = freqs_sq
                mygroup["phi"] = phi
            end

        end

        disp .= current_positions .- reference_data
        disp .*= mass_sqrt 
        disp_mw .= reduce(vcat, eachrow(disp))

        
        #Calculate displacements
        ld = unwrap_coordinates!(ld, reference_data, box_sizes)
        disp = ld.data_storage[!, ["x","y","z"]] .- reference_data[!, ["x","y","z"]]
        for i in eachindex(masses) disp[i] .*= sqrt(masses[i]) end
        disp_mw = reduce(vcat, disp_mw)

        #Convert displacements to mode amplitudes.
        q = phi' * disp_mw;

        #Calculate energy from INMs at timestep i
        mode_potential_order3[:,i] .= (-f0_nmc.*q) .+
                                      0.5.*(freqs_sq .* (q.^2)) .+ 
                                      Array(U_TEP3_n_CUDA(cuK3, cuQ))
        total_eng_INM[i] = @views sum(mode_potential_order3[:,i]) + inma.pot_eng_MD[reference_idx]
        
        dist_btwn = abs(potential_eng_MD[i] - INM_energy)
        total_eng_INM[i] = INM_energy
        
            
        if dist_btwn > max_deviation
            recalculate_INMs = true
            push!(recalculation_idxs, i + 1)
            println(i)
            recalc_counter += 1
        end

    end

    steps_btwn_reset = [recalculation_idxs[i+1] - recalculation_idxs[i] for i in range(1,length(recalculation_idxs)-1)]

    jldopen(joinpath(out_basepath, "INMA.jld2"), "w"; compress = true) do file
        file["steps_btwn_reset"] = steps_btwn_reset
        file["avg_reset_time"] = mean(steps_btwn_reset)
        file["recalculation_idxs"] = recalculation_idxs
        file["max_deviation"] = max_deviation
        file["mode_potential_order3"] = mode_potential_order3
        file["total_eng_INM"] = total_eng_INM
        file["pot_eng_MD"] = inma.pot_eng_MD

    end

    close(dump_file)

    return total_eng_INM, steps_btwn_reset

end
