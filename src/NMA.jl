export NMA

"""
Takes equilibrium positions and calcultes force constants and modal coupling constsants. These values
are to consturct the Taylor Effective Potential using data from a LAMMPS simulation.

 - eq::LammpsDump : Equilibrium data parsed from dump file
 - ld::LammpsDump : Simulation data parsed from dump file
"""
function NMA(eq::LammpsDump, ld::LammpsDump, pair_potential::Potential, potential_eng_MD, masses, out_basepath, T_des, kB)
   
    #Assumes 3D
    box_sizes = [ld.header_data["L_x"][2],ld.header_data["L_y"][2],ld.header_data["L_z"][2]]
    N_modes = 3*length(masses)

    sys = SuperCellSystem(eq.data_storage, masses, box_sizes, "x", "y", "z")

    # Initialize NMs to 3rd Order
    freqs_sq, phi, K3 = get_modal_data(sys, pair_potential)
    
    #Save mode data
    jldopen(joinpath(out_basepath, "NMA.jld2"), "a+") do file
        file["freqs_sq"] = freqs_sq
        file["phi"] = phi
        file["K3"] = K3
    end


    dump_file = open(ld.path, "r")

    initial_data = copy(ld.data_storage)

    #Pre-allocate
    mode_potential_order2 = zeros(N_modes)
    mode_potential_order3 = zeros(N_modes)
    total_eng_NM = zeros(ld.n_samples)

    for i in 1:ld.n_samples

        ld, dump_file = parse_next_timestep!(ld, dump_file) #this prevents CPU parallelization of this code data is stored in this object
        
        #Calculate displacements
        disp = ld.data_storage[!, ["xu","yu","zu"]] .- initial_data[!, ["xu","yu","zu"]]
        for col in eachcol(disp) col .*= sqrt.(masses) end
        disp_mw = reduce(vcat, Matrix(disp))

        #Convert displacements to mode amplitudes.
        q_anharmonic = phi' * disp_mw;

        #Calculate energy from INMs at timestep i
        mode_potential_order2 .= 0.5.*(freqs_sq .* (q_anharmonic.^2))
        mode_potential_order3 .= mode_potential_order2 .+ U_TEP3_n.(Ref(K3), Ref(q_anharmonic), range(1,N_modes))
        total_eng_NM[i]  = sum(mode_potential_order3) + potential_eng_MD[firstindex(potential_eng_MD)]
               
        #Append energies to file
        jldopen(joinpath(out_basepath, "NMA.jld2"), "a+") do file
            file["/Step $i/mode_potential_order2"] = mode_potential_order2
            file["/Step $i/mode_potential_order3"] = mode_potential_order3
        end
            

    end

    jldopen(joinpath(out_basepath, "NMA.jld2"), "a+") do file
        file["total_eng_NM"] = total_eng_NM
        file["potential_eng_MD"] = potential_eng_MD
        file["cv_MD_total"] = var(potential_eng_MD)/(kB*T_des*T_des)
        file["cv_TEP_total"] = var(total_eng_NM)/(kB*T_des*T_des)
    end

    close(dump_file)

end
