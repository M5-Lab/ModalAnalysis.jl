using Revise
using Pkg
Pkg.activate(raw"C:\Users\ejmei\repos\ModalAnalysis.jl")
Pkg.instantiate()
Pkg.resolve()

using ModalAnalysis
using Unitful
using ForceConstants

#Load Dump File
dump_path = raw"C:\Users\ejmei\repos\ModalAnalysis.jl\examples\LJ_FCC_2UC\dump.atom"
ld = LammpsDump(dump_path);

#Load Thermo Data & Masses
thermo_data = ThermoData()
masses = load_masses()

#Replicate potential used in LAMMSP
pot = LJ(3.4u"Å", 0.24037u"kcal * mol^-1", 8.5u"Å")

out_path = raw"C:\Users\ejmei\repos\ModalAnalysis.jl\examples\LJ_FCC_2UC\"

T_des = 10.0u"K"
max_deviation = 0.01*mean(potential_eng_MD)

#Dump needs xu,yu,zu
NMA(ld, pot, potential_eng_MD, masses, out_path, T_des, kB)

#Dump needs x, y, z, ix, iy, iz, fx, fy, fz
INMA(ld, pot, potential_eng_MD, masses, max_deviation, out_path, T_des, kB)