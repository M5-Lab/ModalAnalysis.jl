using Pkg
Pkg.activate("C:/Users/ejmei/repos/ModalAnalysis.jl")

using ModalAnalysis
using ForceConstants
using SimpleCrystals
using Unitful
using StaticArrays


crys = FCC(5.43u"Å", :Ar, SVector(4,4,4))
sys = SuperCellSystem(crys);

pot = LJ(3.4u"Å", 0.24037u"kcal/mol", 8.5u"Å")
calc = AnalyticalCalculator(1e-8, pot.r_cut)

temps = [300]
n_configs = 1
n_iters = 1
outpath = "C:/Users/ejmei/Box/Research/Projects/ThermalCond/sc_loop_data"

configs = SelfConsistentLoopJob(sys, temps, calc, pot, n_configs, n_iters, outpath)