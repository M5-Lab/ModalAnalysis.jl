using Revise
using Pkg
Pkg.activate(raw"C:\Users\ejmei\repos\ModalAnalysis.jl")
Pkg.instantiate()
Pkg.resolve()

using ModalAnalysis
using Unitful
using ForceConstants
using DelimitedFiles

###############
# NMA EXAMPLE #
###############
base_path = raw"C:\Users\ejmei\repos\ModalAnalysis.jl\examples\LJ_FCC_3UC\NMA"
equilibrium_data_path = joinpath(base_path, "equilibrium.atom")
dump_path = joinpath(base_path, "dump.atom")
thermo_path = joinpath(base_path,"thermo_data.txt")

eq = LammpsDump(equilibrium_data_path);
parse_timestep!(eq, 1)
masses = get_col(eq, "mass")

ld = LammpsDump(dump_path);

#Load Thermo Data & Masses
pe_col = 3
thermo_data = readdlm(thermo_path, skipstart = 2);
potential_eng_MD = thermo_data[:,pe_col]

#Replicate potential used in LAMMSP
pot = LJ(3.4, 0.24037, 8.5);

out_path = base_path

T_des = 10.0

#Dump needs xu,yu,zu
kB = ustrip(u"kcal * mol^-1 * K^-1", Unitful.k*Unitful.Na)
ProfileView.@profview NMA(eq, ld, pot, potential_eng_MD, masses, out_path, T_des, kB)


###############
# INMA EXAMPLE #
###############
# max_deviation = 0.01*mean(potential_eng_MD)

#Dump needs x, y, z, ix, iy, iz, fx, fy, fz
# INMA(ld, pot, potential_eng_MD, masses, max_deviation, out_path, T_des, kB)