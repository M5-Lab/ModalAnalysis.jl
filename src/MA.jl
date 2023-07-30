export NormalModeAnalysis, InstantaneousNormalModeAnalysis

"""
ModalAnalysis.jl expects the following file structure for a 
`NormalModeAnalysis` or `InstantaneousNormalModeAnalysis`.
    └── <simulation_folder>
            ├── dump.atom
            ├── equilibrium.atom
            ├── thermo_data.txt
"""
abstract type ModalAnalysisAlgorithm end

#TODO: Move to conf file
const global TEMPERATURE_COL::Int64 = 2
const global POTENTIAL_ENG_COL::Int64 = 3
const global FC_TOL::Float64 = 1e-12
const global DUMP_NAME::String = "dump.atom"
const global EQ_NAME::String = "equilibrium.atom"
const global THERMO_NAME::String = "thermo_data.txt"

struct NormalModeAnalysis{T,M} <: ModalAnalysisAlgorithm 
    simulation_folder::String
    potential::Potential
    temperature::T
    atom_masses::Vector{M}
    pot_eng_MD::Vector{Float64}
    eq::LammpsDump
    ld::LammpsDump
    sys::SuperCellSystem
end

function NormalModeAnalysis(simulation_folder, pot, temperature)
    atom_masses, pot_eng_MD, T_avg, eq, ld = parse_simulation_data(simulation_folder)
    check_temp(nma, T_avg)

    @assert issubset(["xu","yu","zu"], eq.header_data["fields"]) "NMA equilibrium data needs xu, yu and zu fields"
    @assert issubset(["xu","yu","zu"], ld.header_data["fields"]) "NMA dump data needs xu, yu and zu fields"

    box_sizes = [ld.header_data["L_x"][2], ld.header_data["L_y"][2], ld.header_data["L_z"][2]]
    sys = SuperCellSystem(eq.data_storage, atom_masses, box_sizes, "xu", "yu", "zu")

    return NormalModeAnalysis{typeof(temperature), eltype(atom_masses)}(
        simulation_folder, pot, temperature, atom_masses, pot_eng_MD, eq, ld, sys)
end

struct InstantaneousNormalModeAnalysis{T,M} <: ModalAnalysisAlgorithm
    simulation_folder::String
    potential::Potential
    temperature::T
    atom_masses::Vector{M}
    pot_eng_MD::Vector{Float64}
    eq::LammpsDump
    ld::LammpsDump
    sys::SuperCellSystem
end

function InstantaneousNormalModeAnalysis(simulation_folder, pot, temperature)
    atom_masses, pot_eng_MD, T_avg, eq, ld = parse_simulation_data(simulation_folder)
    check_temp(temperature, T_avg)

    box_sizes = [ld.header_data["L_x"][2], ld.header_data["L_y"][2], ld.header_data["L_z"][2]]
    sys = SuperCellSystem(eq.data_storage, atom_masses, box_sizes, "xu", "yu", "zu")

    return InstantaneousNormalModeAnalysis{typeof(temperature), eltype(atom_masses)}(
        simulation_folder, pot, temperature, atom_masses, pot_eng_MD, eq, ld, sys)
end


function parse_simulation_data(path::String)

    equilibrium_data_path = joinpath(path, EQ_NAME)
    dump_path = joinpath(path, DUMP_NAME)
    thermo_path = joinpath(path, THERMO_NAME)

    eq = LammpsDump(equilibrium_data_path);
    parse_timestep!(eq, 1)
    atom_masses = get_col(eq, "mass")

    ld = LammpsDump(dump_path);

    #Load Thermo Data & Masses
    thermo_data = readdlm(thermo_path, skipstart = 2);
    potential_eng_MD = thermo_data[:,POTENTIAL_ENG_COL]
    temps_MD = thermo_data[:, TEMPERATURE_COL]

    T_avg = mean(temps_MD)

    return atom_masses, potential_eng_MD, T_avg, eq, ld

end

function check_temp(T, T_actual::Float64)
    if !isapprox(T, T_actual, atol = 1e-2)
        @warn "ModalAnalysisAlgorithm expected temperature: $(T) but MD simulation average was $(T_actual)"
    end
end

# This function is bottle neck, specifically F3_2_K3
function get_modal_data(ma::ModalAnalysisAlgorithm)
    dynmat = dynamicalMatrix(ma.sys, ma.potential, FC_TOL)
    freqs_sq, phi = get_modes(dynmat)

    Ψ = third_order_IFC(ma.sys, ma.potential, FC_TOL);
    @info "IFC3 calculation complete"
    Ψ = mass_weight_third_order!(Ψ, masses(sys))

    cuΨ = CuArray(Ψ.values); cuPhi = CuArray(Float32.(phi))
    K3 = mcc3(cuΨ, cuPhi, tol = FC_TOL);

    @info "MCC3 calculation complete"
    return freqs_sq, phi, dynmat, Ψ, K3
end

function get_modal_data(ma::ModalAnalysisAlgorithm, mcc_block_size::Integer)
    dynmat = dynamicalMatrix(ma.sys, ma.potential, FC_TOL)
    freqs_sq, phi = get_modes(dynmat)

    Ψ = third_order_IFC(ma.sys, ma.potential, FC_TOL);
    @info "IFC3 calculation complete"
    Ψ = mass_weight_third_order!(Ψ, masses(sys))

    cuΨ = CuArray{Float32}(Ψ.values); cuPhi = CuArray{Float32}(phi)
    K3 = mcc3(cuΨ, cuPhi, mcc_block_size, tol = FC_TOL);

    @info "MCC3 calculation complete"
    return freqs_sq, phi, dynmat, Ψ, K3
end

 
# function start_write_thread(results::RemoteChannel)

#     Thread.@spawn begin
#         take!(results)
        
#     end

# end