export NormalModeAnalysis, InstantaneousNormalModeAnalysis, get_modal_data, get_sys

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
const global KINETIC_ENG_COL::Int64 = 4
const global FC_TOL::Float64 = 1e-12
const global DUMP_NAME::String = "dump.atom"
const global EQ_POSN_NAME::String = "equilibrium.atom"
const global EQ_ENG_NAME::String = "equilibrium.energies"
const global THERMO_NAME::String = "thermo_data.txt"

struct NormalModeAnalysis{T,M} <: ModalAnalysisAlgorithm 
    simulation_folder::String
    temperature::T
    atom_masses::Vector{M}
    pot_eng_MD::Vector{Float64}
    eq_pot_eng::Float64
    eq::LammpsDump
    ld::LammpsDump
    sys::SuperCellSystem
end

function NormalModeAnalysis(simulation_folder, temperature)
    atom_masses, pot_eng_MD, T_avg, eq, ld = parse_simulation_data(simulation_folder)
    eq_pot_eng = parse_eq_energy(simulation_folder)
    check_temp(temperature, T_avg)
    #T_avg *= unit(temperature)

    unrolled_coords = issubset(["xu","yu","zu"], eq.header_data["fields"])
    normal_coords = issubset(["x","y","z"], eq.header_data["fields"])

    @assert (unrolled_coords || normal_coords) "NMA equilibrium data needs xu, yu and zu or x,y,z fields"
    @assert issubset(["xu","yu","zu"], ld.header_data["fields"]) "NMA dump data needs xu, yu and zu fields"

    box_sizes = [ld.header_data["L_x"][2], ld.header_data["L_y"][2], ld.header_data["L_z"][2]]
    if normal_coords
        sys = SuperCellSystem(eq.data_storage, atom_masses, box_sizes, "x", "y", "z")
    else
        sys = SuperCellSystem(eq.data_storage, atom_masses, box_sizes, "xu", "yu", "zu")
    end

    return NormalModeAnalysis{typeof(T_avg), eltype(atom_masses)}(
        simulation_folder, T_avg, atom_masses, pot_eng_MD, eq_pot_eng, eq, ld, sys)
end

get_sys(nma::NormalModeAnalysis) = nma.sys


mutable struct InstantaneousNormalModeAnalysis{T,M} <: ModalAnalysisAlgorithm
    const simulation_folder::String
    const temperature::T
    const atom_masses::Vector{M}
    const pot_eng_MD::Vector{Float64}
    const ld::LammpsDump
    reference_sys::SuperCellSystem
end

function InstantaneousNormalModeAnalysis(simulation_folder, temperature)
    atom_masses, pot_eng_MD, T_avg, eq, ld = parse_simulation_data(simulation_folder)
    check_temp(temperature, T_avg)
    #T_avg *= unit(temperature)

    @assert issubset(["x","y","z","fx","fy","fz","ix","iy","iz"], ld.header_data["fields"]) "INMA dump data needs x,y,z,fx,fy,fz,ix,iy,iz fields"

    box_sizes = [ld.header_data["L_x"][2], ld.header_data["L_y"][2], ld.header_data["L_z"][2]]
    sys = SuperCellSystem(eq.data_storage, atom_masses, box_sizes, "x", "y", "z")

    return InstantaneousNormalModeAnalysis{typeof(T_avg), eltype(atom_masses)}(
        simulation_folder, T_avg, atom_masses, pot_eng_MD, ld, sys)
end

get_sys(inma::InstantaneousNormalModeAnalysis) = inma.reference_sys

"""
Updates the `reference_sys` parmaater of an `InstantaneousNormalModeAnalysis` object so that
the positions stored match those in `ld.data_storage`.
"""
function update_reference_sys!(inma::InstantaneousNormalModeAnalysis)
    box_sizes = [inma.ld.header_data["L_x"][2], inma.ld.header_data["L_y"][2], inma.ld.header_data["L_z"][2]]
    inma.reference_sys = SuperCellSystem(inma.ld.data_storage, inma.atom_masses, box_sizes, "x", "y", "z")
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
    kinetic_eng_MD = thermo_data[:,KINETIC_ENG_COL]
    temps_MD = thermo_data[:, TEMPERATURE_COL]

    T_avg = mean(temps_MD)

    # Check that assumption of Var(U) + Var(K) = Var(K + U) is valid
    tot_var = var(potential_eng_MD + kinetic_eng_MD)
    covar = 2*cov(potential_eng_MD, kinetic_eng_MD)

    if covar > 0.01*tot_var
        @warn "Covaraicne accounts for more than 1% of heat capacity. Var(U + K) linearly assumption
         could be invalid. Covaraince: $(covar) Total Variance: $(tot_var)"
    end

    return atom_masses, potential_eng_MD, T_avg, eq, ld

end

function parse_eq_energy(path::String)
    eq_eng_path = joinpath(path, EQ_ENG_NAME)
    eq_pot_eng = (readdlm(eq_eng_path, comments = true))[POTENTIAL_ENG_COL]
    return eq_pot_eng
end

function check_temp(T, T_actual::Float64)
    if !isapprox(T, T_actual, atol = 1)
        @warn "ModalAnalysisAlgorithm expected temperature: $(T) but MD simulation average was $(T_actual)"
    end
end

"""
Frequencies, eigenvectors, dynamical matrix and MCC3
"""
function get_modal_data(ma::ModalAnalysisAlgorithm)
    s = get_sys(ma)
    dynmat = dynamicalMatrix(s, ma.potential, FC_TOL)
    freqs_sq, phi = get_modes(dynmat)

    Ψ = third_order_IFC(s, ma.potential, FC_TOL); #&this probably slows down INMs most

    @info "IFC3 calculation complete"
    Ψ = mass_weight_third_order!(Ψ, masses(s)) #&can I make this float32 throughout?

    cuΨ = CuArray{Float32}(Ψ.values); cuPhi = CuArray(Float32.(phi))
    K3 = mcc3!(cuΨ, cuPhi);

    @info "MCC3 calculation complete"
    return freqs_sq, phi, dynmat, K3
end

function get_modal_data(ma::ModalAnalysisAlgorithm, mcc_block_size::Integer)
    s = get_sys(ma)
    dynmat = dynamicalMatrix(s, ma.potential, FC_TOL)
    freqs_sq, phi = get_modes(dynmat)

    Ψ = third_order_IFC(s, ma.potential, FC_TOL);
    @info "IFC3 calculation complete"
    Ψ = mass_weight_third_order!(Ψ, masses(s))

    cuΨ = CuArray{Float32}(Ψ.values); cuPhi = CuArray{Float32}(phi)
    K3 = mcc3(cuΨ, cuPhi, mcc_block_size);

    @info "MCC3 calculation complete"
    return freqs_sq, phi, dynmat, K3
end
