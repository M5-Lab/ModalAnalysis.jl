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
function get_modal_data(sys, potential; tol = 1e-12)
    dynmat = dynamicalMatrix(sys, potential, tol)
    freqs_sq, phi = get_modes(dynmat)

    Ψ = third_order_IFC(sys, potential, tol);
    Ψ_sparse_mw = mass_weight_sparsify_third_order(Ψ, masses(sys))

    # This only gives upper 1/6th of K3 Tensor
    K3 = F3_2_K3(Ψ_sparse_mw, phi, length(r0), tol);
    
    return freqs_sq, phi, K3
end

"""
Takes the output from ForceConstants.jl and splits up
"""
function sort_mcc_by_mode(K3)

end

