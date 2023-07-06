"""
This file includes functions common to both Instantaneous Normal Mode Analysis (INMA.jl) 
and Normal Mode Analysis (NMA.jl).
"""

abstract type WorkSplittingStrategy end
struct SplitByParameter <: WorkSplittingStrategy end # 1 dump file per GPU
struct SplitByData <: WorkSplittingStrategy end # Multiple GPU per dump file

abstract type ModalAnalysisAlgorithm end
struct NormalModeAnalysis{M} <: ModalAnalysisAlgorithm 
    dump_file_paths::Vector{String}
    thermo_data_file_paths::Vector{String}
    potential::Potential
    masses::Vector{M}
end

function NormalModeAnalysis(dump_file_paths, thermo_data_file_paths)


end

struct InstantaneousNormalModeAnalysis{M} <: ModalAnalysisAlgorithm
    dump_file_paths::Vector{String}
    thermo_data_file_paths::Vector{String}
    potential::Potential
    masses::Vector{M}
end

function InstantaneousNormalModeAnalysis(dump_file_paths, thermo_data_file_paths)


end

# This function is bottle neck, specifically F3_2_K3
function calculate_modes(r0, potential, masses, L; tol = 1e-12)
    Φ = second_order_IFC(r0, potential, length(r0), L, tol);
    Ψ = third_order_IFC(r0, potential, length(r0), L, tol);
    Ψ_sparse_mw = to_sparse_mw(Ψ, masses)
    dynmat = IFC2_dynmat(Φ, masses)
    freqs, phi = parseDynmat(dynmat)
    K3 = F3_2_K3(Ψ_sparse_mw, phi, length(r0), tol);
    
    return freqs, phi, K3
end

function split_work(dump_file_paths::Vector{String}, wss::SplitByParameter)


end

function split_work(dump_file_paths::Vector{String}, wss::SplitByData)


end