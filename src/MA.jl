"""
This file includes functions common to both Instantaneous Normal Mode Analysis (INMA.jl) 
and Normal Mode Analysis (NMA.jl).
"""

# abstract type WorkSplittingStrategy end
# struct SplitByParameter <: WorkSplittingStrategy end # 1 dump file per GPU
# struct SplitByData <: WorkSplittingStrategy end # Multiple GPU per dump file

# abstract type ModalAnalysisAlgorithm end
# struct NormalModeAnalysis{M} <: ModalAnalysisAlgorithm 
#     dump_file_paths::Vector{String}
#     thermo_data_file_paths::Vector{String}
#     potential::Potential
#     masses::Vector{M}
# end

# function NormalModeAnalysis(dump_file_paths, thermo_data_file_paths)


# end

# struct InstantaneousNormalModeAnalysis{M} <: ModalAnalysisAlgorithm
#     dump_file_paths::Vector{String}
#     thermo_data_file_paths::Vector{String}
#     potential::Potential
#     masses::Vector{M}
# end

# function InstantaneousNormalModeAnalysis(dump_file_paths, thermo_data_file_paths)


# end


# This function is bottle neck, specifically F3_2_K3
function get_modal_data(sys, potential; gpu_device_id::Integer = 0, tol = 1e-12)
    dynmat = dynamicalMatrix(sys, potential, tol)
    freqs_sq, phi = get_modes(dynmat)

    Ψ = third_order_IFC(sys, potential, tol);
    @info "IFC3 calculation complete"
    Ψ = mass_weight_third_order!(Ψ, masses(sys))

    cuΨ = CuArray(Ψ.values); cuPhi = CuArray(Float32.(phi))
    K3 = mcc3(cuΨ, cuPhi, tol = tol, gpu_id = gpu_device_id);

    @info "MCC3 calculation complete"
    return freqs_sq, phi, dynmat, Ψ, K3
end

function get_modal_data(sys, potential, mcc_block_size::Integer; gpu_device_id::Integer = 0, tol = 1e-12)
    dynmat = dynamicalMatrix(sys, potential, tol)
    freqs_sq, phi = get_modes(dynmat)

    Ψ = third_order_IFC(sys, potential, tol);
    @info "IFC3 calculation complete"
    Ψ = mass_weight_third_order!(Ψ, masses(sys))

    cuΨ = CuArray{Float32}(Ψ.values); cuPhi = CuArray{Float32}(phi)
    K3 = mcc3(cuΨ, cuPhi, mcc_block_size, tol = tol, gpu_id = gpu_device_id);

    @info "MCC3 calculation complete"
    return freqs_sq, phi, dynmat, Ψ, K3
end

 
# function start_write_thread(results::RemoteChannel)

#     Thread.@spawn begin
#         take!(results)
        
#     end

# end