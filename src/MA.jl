export NormalModeAnalysis, InstantaneousNormalModeAnalysis, get_modal_data

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
const global POTENTIAL_ENG_COL::Int64 = 3 #for both thermo_data and eq.engs
const global FC_TOL::Float64 = 1e-12
const global DUMP_NAME::String = "dump.atom"
const global EQ_POSN_NAME::String = "equilibrium.atom"
const global EQ_ENG_NAME::String = "equilibrium.energies"
const global THERMO_NAME::String = "thermo_data.txt"

struct NormalModeAnalysis{T,M} <: ModalAnalysisAlgorithm 
    simulation_folder::String
    potential::Potential
    temperature::T
    atom_masses::Vector{M}
    pot_eng_MD::Vector{Float64}
    eq_pot_eng::Float64
    eq::LammpsDump
    ld::LammpsDump
    sys::SuperCellSystem
end

function NormalModeAnalysis(simulation_folder, pot, temperature)
    atom_masses, pot_eng_MD, T_avg, eq, ld = parse_simulation_data(simulation_folder)
    eq_pot_eng = parse_eq_energy(simulation_folder)
    check_temp(temperature, T_avg)

    @assert issubset(["xu","yu","zu"], eq.header_data["fields"]) "NMA equilibrium data needs xu, yu and zu fields"
    @assert issubset(["xu","yu","zu"], ld.header_data["fields"]) "NMA dump data needs xu, yu and zu fields"

    box_sizes = [ld.header_data["L_x"][2], ld.header_data["L_y"][2], ld.header_data["L_z"][2]]
    sys = SuperCellSystem(eq.data_storage, atom_masses, box_sizes, "xu", "yu", "zu")

    return NormalModeAnalysis{typeof(temperature), eltype(atom_masses)}(
        simulation_folder, pot, temperature, atom_masses, pot_eng_MD, eq_pot_eng, eq, ld, sys)
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
    sys = SuperCellSystem(eq.data_storage, atom_masses, box_sizes, "x", "y", "z")

    return InstantaneousNormalModeAnalysis{typeof(temperature), eltype(atom_masses)}(
        simulation_folder, pot, temperature, atom_masses, pot_eng_MD, eq, ld, sys)
end


function parse_simulation_data(path::String)

    eq_posn_path = joinpath(path, EQ_POSN_NAME)
    dump_path = joinpath(path, DUMP_NAME)
    thermo_path = joinpath(path, THERMO_NAME)

    eq = LammpsDump(eq_posn_path);
    parse_timestep!(eq, 1)
    atom_masses = get_col(eq, "mass")

    ld = LammpsDump(dump_path);

    #Load Thermo Data & Masses
    thermo_data = readdlm(thermo_path, comments = true);
    potential_eng_MD = thermo_data[:,POTENTIAL_ENG_COL]
    temps_MD = thermo_data[:, TEMPERATURE_COL]

    T_avg = mean(temps_MD)

    return atom_masses, potential_eng_MD, T_avg, eq, ld

end

function parse_eq_energy(path::String)
    eq_eng_path = joinpath(path, EQ_ENG_NAME)
    eq_pot_eng = (readdlm(eq_eng_path, comments = true))[POTENTIAL_ENG_COL]
    return eq_pot_eng
end

function check_temp(T, T_actual::Float64)
    if !isapprox(T, T_actual, atol = 1e-1)
        @warn "ModalAnalysisAlgorithm expected temperature: $(T) but MD simulation average was $(T_actual)"
    end
end

"""
Frequencies, eigenvectors, dynamical matrix and MCC3
"""
function get_modal_data(ma::ModalAnalysisAlgorithm)
    dynmat = dynamicalMatrix(ma.sys, ma.potential, FC_TOL)
    freqs_sq, phi = get_modes(dynmat)

    Ψ = third_order_IFC(ma.sys, ma.potential, FC_TOL); #&this probably slows down INMs most

    @info "IFC3 calculation complete"
    Ψ = mass_weight_third_order!(Ψ, masses(ma.sys)) #&can I make this float32 throughout?

    cuΨ = CuArray{Float32}(Ψ.values); cuPhi = CuArray(Float32.(phi))
    K3 = mcc3(cuΨ, cuPhi, FC_TOL);

    @info "MCC3 calculation complete"
    return freqs_sq, phi, dynmat, K3
end

function get_modal_data(ma::ModalAnalysisAlgorithm, mcc_block_size::Integer)
    dynmat = dynamicalMatrix(ma.sys, ma.potential, FC_TOL)
    freqs_sq, phi = get_modes(dynmat)

    Ψ = third_order_IFC(ma.sys, ma.potential, FC_TOL);
    @info "IFC3 calculation complete"
    Ψ = mass_weight_third_order!(Ψ, masses(ma.sys))

    cuΨ = CuArray{Float32}(Ψ.values); cuPhi = CuArray{Float32}(phi)
    K3 = mcc3(cuΨ, cuPhi, mcc_block_size, FC_TOL);

    @info "MCC3 calculation complete"
    return freqs_sq, phi, dynmat, K3
end
