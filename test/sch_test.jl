using Pkg
Pkg.activate("C:/Users/ejmei/repos/ModalAnalysis.jl")

using ModalAnalysis
using ForceConstants
using SimpleCrystals
using Unitful
using StaticArrays


crys = FCC(5.2468u"Å", :Ar, SVector(4,4,4))
sys = SuperCellSystem(crys);

pot = LJ(3.4u"Å", 0.24037u"kcal/mol", 8.5u"Å")
calc = AnalyticalCalculator(1e-8, pot.r_cut)


temps = [80]
n_configs = 200
n_iters = 10
outpath = "C:/Users/ejmei/Box/Research/Projects/ThermalCond/sc_loop_data"



configs = SelfConsistentLoopJob(sys, temps, calc, pot, n_configs, n_iters, outpath)


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