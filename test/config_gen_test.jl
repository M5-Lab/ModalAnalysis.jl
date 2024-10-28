using Pkg
Pkg.activate("C:/Users/ejmei/repos/ModalAnalysis.jl")
using Revise

using ModalAnalysis
using ForceConstants
using SimpleCrystals
using Unitful
using StaticArrays
using JLD2


crys = FCC(5.2468u"Å", :Ar, SVector(4,4,4))
sys = SuperCellSystem(crys);

pot = LJ(3.4u"Å", 0.24037u"kcal/mol", 8.5u"Å")
calc = AnalyticalCalculator(1e-8, pot.r_cut)


temps = [80]
n_configs = 50000
outpath = "C:/Users/ejmei/Box/Research/Projects/ThermalCond/sc_loop_data"

dynmat = dynamical_matrix(sys, pot, calc)
freqs_sq, phi = get_modes(dynmat, 3)
freqs = sqrt.(freqs_sq);


ifc_path = raw"Z:\emeitz\Data\NMA\LJ\LJ_FCC_Qual\TDEP\TEP_TDEP_80K.jld2"
# ifc_path = raw"Z:\emeitz\Data\ForceConstants\AvgINM_LJ\AvgIFC_LJ_80K_CLEANED.jld2"
freqs_sq, phi = load(ifc_path, "freqs_sq", "phi");
freqs = sqrt.(freqs_sq);

configs, energies = GenerateConfigs(sys, temps, freqs, phi, n_configs, energy_unit(pot);
                             mode = :classical, pot = pot);


function write_xyz(positions, outpath, N_atoms, D = 3)

    open(outpath, "w") do f
        for config in eachcol(positions)
            println(f, N_atoms)
            println(f, "COMMENT LINE")
            for i in 1:N_atoms
                println(f, "X $(config[D*(i-1) + 1]) $(config[D*(i-1) + 2]) $(config[D*(i-1) + 3])")
            end
        end
    end

end

write_xyz(configs[1].result.value.configs, joinpath(outpath, "test.xyz"), 256)