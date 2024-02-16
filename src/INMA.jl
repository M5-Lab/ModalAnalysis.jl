export AbsoluteDistanceMetric, VarianceMetric, TimestepReset, run

abstract type DeviationMetric end

struct AbsoluteDistanceMetric <: DeviationMetric
    allowed_absolute_difference::Float64
    AbsoluteDistanceMetric(aad) = aad <= 0 ? error("Must be positive") : new(aad)
end

struct TimestepReset <: DeviationMetric
    every_n_steps::Integer
end

struct LeveneMetric <: DeviationMetric
    levene_test_pval::Float64
    initial_steps::Integer
end

function check_energy_deviation(MD_energy, INM_energy, current_idx, last_reset_idx, dm::AbsoluteDistanceMetric)
    return (abs(MD_energy[current_idx] - INM_energy[current_idx]) > dm.allowed_absolute_difference)
end

function check_energy_deviation(MD_energy, INM_energy, current_idx, last_reset_idx, dm::TimestepReset)
    return (current_idx - last_reset_idx) >= dm.every_n_steps
end


function check_energy_deviation(MD_energy, INM_energy, current_idx, last_reset_idx, dm::LeveneMetric)
    if current_idx - last_reset_idx > dm.initial_steps
        lt = LeveneTest(view(MD_energy, last_reset_idx:current_idx),view(INM_energy, last_reset_idx:current_idx))
        return pvalue(lt) < dm.levene_test_pval #reject null hypo that var are the same
    else
        return false
    end
end



###################

"""
    run(nma::NormalModeAnalysis, dm::DeviationMetric)
    run(nma::NormalModeAnalysis, mcc_block_size::Integer, dm::DeviationMetric)
"""
function run(inma::InstantaneousNormalModeAnalysis, dm::DeviationMetric, calc::ForceConstantCalculator)
    dynmat = dynamical_matrix(inma.reference_sys, inma.potential, calc)
    freqs_sq, _ = get_modes(dynmat)
    N_modes = length(freqs_sq)
    INMA_loop(inma, inma.simulation_folder, dm, nothing, N_modes)
end

function run(inma::InstantaneousNormalModeAnalysis, mcc_block_size::Integer, 
    dm::DeviationMetric, calc::ForceConstantCalculator)
    
    dynmat = dynamical_matrix(inma.reference_sys, inma.potential, calc)
    freqs_sq, _ = get_modes(dynmat)
    N_modes = length(freqs_sq)
    INMA_loop(inma, inma.simulation_folder, dm, mcc_block_size, N_modes)
end

function calculate_INMs(inma::InstantaneousNormalModeAnalysis, mcc_block_size, mass_sqrt)

    #Update the positions used to calculate force constants
    update_reference_sys!(inma)

    # Initialize INMs to 3rd Order
    #TODO write version that keeps K3 on GPU (only possible if it fits)
    #TODO Re-use K3 storage if it has to go on CPU?
    if mcc_block_size === nothing
        freqs_sq, phi, dynmat, K3 = get_modal_data(inma)
    else
        freqs_sq, phi, dynmat, K3 = get_modal_data(inma, mcc_block_size)
    end
    cuK3 = CuArray{Float32}(K3)


    #Set new reference positions to calculate displacements 
    reference_data = deepcopy(inma.ld.data_storage)
    # reference_data = zero.(inma.ld.data_storage)

    forces = Matrix(reference_data[!,["fx","fy","fz"]])
    f0 = reduce(vcat, eachrow(forces) ./ mass_sqrt)

    f0_nmc = (phi') * f0

    return f0_nmc, freqs_sq, phi, cuK3, reference_data
end


function INMA_loop(inma::InstantaneousNormalModeAnalysis, out_path::String,
        dm::DeviationMetric, mcc_block_size::Union{Integer, Nothing}, N_modes::Integer)
    
    N_atoms = length(inma.atom_masses)
    mass_sqrt = sqrt.(inma.atom_masses)
    #Assumes 3D
    box_sizes = [inma.ld.header_data["L_x"][2],inma.ld.header_data["L_y"][2],inma.ld.header_data["L_z"][2]]

    # Set initial INM state
    recalculate_INMs = true
    recalculation_idxs = [1]
    reference_idx = 1

    #Pre-allocate arrays
    mode_potential_order3 = zeros(N_modes, inma.ld.n_samples)
    total_eng_INM = zeros(inma.ld.n_samples)
    disp = zeros(N_atoms,3)
    disp_mw = zeros(N_modes)
    q = zeros(Float32,N_modes)
    cuQ = CUDA.zeros(N_modes)

    dump_file = open(inma.ld.path, "r")
    f0_nmc, freqs_sq, phi, cuK3, reference_data = nothing, nothing, nothing, nothing, nothing

 
    recalc_counter = 1
    for i in 1:inma.ld.n_samples
        #Parse data from dump file into inma.ld.data_storage
        parse_next_timestep!(inma.ld, dump_file)

        if recalculate_INMs
            #& probably should re-use this storage instead of re-allocating
            f0_nmc, freqs_sq, phi, cuK3, reference_data = calculate_INMs(inma, mcc_block_size, mass_sqrt) 

            recalculate_INMs = false
            reference_idx = i

            #Create new group
            group_name = "INM$(recalc_counter)"
            jldopen(joinpath(out_path, "ModeData.jld2"), "a+") do file
                mygroup = JLD2.Group(file, group_name)
                mygroup["freqs_sq"] = freqs_sq
                mygroup["phi"] = phi
            end

        end

        #Calculate displacements
        unwrap_coordinates!(inma.ld, reference_data, box_sizes)
        disp .= Matrix(inma.ld.data_storage[!, ["x","y","z"]] .- reference_data[!, ["x","y","z"]])
        disp_mw .= reduce(vcat, eachrow(disp) .* mass_sqrt)

        #Convert displacements to mode amplitudes.
        mul!(q, phi', disp_mw)
        copyto!(cuQ, q)

        #Calculate energy from INMs at timestep i
        mode_potential_order3[:,i] .= (-f0_nmc.*q) .+
                                      0.5.*(freqs_sq .* (q.^2)) .+ 
                                      Array(U_TEP3_n_CUDA(cuK3, cuQ))
        total_eng_INM[i] = @views sum(mode_potential_order3[:,i]) + inma.pot_eng_MD[reference_idx]
        
        if check_energy_deviation(inma.pot_eng_MD, total_eng_INM, i, reference_idx, dm)
            recalculate_INMs = true
            push!(recalculation_idxs, i + 1)
            println(i)
            recalc_counter += 1
        end

    end

    steps_btwn_reset = [recalculation_idxs[i+1] - recalculation_idxs[i] for i in range(1,length(recalculation_idxs)-1)]

    jldopen(joinpath(out_path, "ModeEnergies.jld2"), "w"; compress = true) do file
        file["steps_btwn_reset"] = steps_btwn_reset
        file["avg_reset_time"] = mean(steps_btwn_reset)
        file["recalculation_idxs"] = recalculation_idxs
        # file["deviation_alg"] = dm #*not sure this works
        file["mode_potential_order3"] = mode_potential_order3
        file["total_eng_INM"] = total_eng_INM
        file["pot_eng_MD"] = inma.pot_eng_MD

    end

    close(dump_file)

    return total_eng_INM, steps_btwn_reset

end
