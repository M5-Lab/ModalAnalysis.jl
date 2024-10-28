using Pkg
Pkg.activate("C:/Users/ejmei/repos/ModalAnalysis.jl")

using ModalAnalysis
using ForceConstants
using SimpleCrystals
using Unitful
using StaticArrays
using JLD2


crys = FCC(5.2468u"Å", 39.95u"g/mol", SVector(4,4,4))
sys = SuperCellSystem(crys);
pot = LJ(3.4u"Å", 0.24037u"kcal/mol", 8.5u"Å")
calc = AnalyticalCalculator(1e-8, pot.r_cut)


# crys = Diamond(5.43u"Å", 28.85u"u", SVector(3,3,3))
# sys = SuperCellSystem(crys);
# pot = StillingerWeberSilicon()
# calc = AutoDiffCalculator(1e-8, pot.r_cut)


temps = [10,20,30,40,50,60,70,80]*u"K"
# temps = [100, 700]*u"K"
n_configs = 500
n_iters = 10


outpath = (T) -> joinpath("C:/Users/ejmei/Desktop", "sch_test_LJ_$(ustrip(T))K.jld2")
# ifc3 = third_order(sys, pot, calc)

sch_results = SelfConsistentLoopJob(sys, temps, calc, pot, n_configs, n_iters, outpath;
                                        mode = :classical, save_every = 2)





# function write_xyz(positions, outpath, N_atoms, D = 3)

#     open(outpath, "w") do f
#         for config in eachcol(positions)
#             println(f, N_atoms)
#             println(f, "COMMENT LINE")
#             for i in 1:N_atoms
#                 println(f, "X $(config[D*(i-1) + 1]) $(config[D*(i-1) + 2]) $(config[D*(i-1) + 3])")
#             end
#         end
#     end

# end

# write_xyz(configs[1].result.value.configs, joinpath(outpath, "test.xyz"), 256) 