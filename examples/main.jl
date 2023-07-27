using Revise
using Pkg
Pkg.activate(raw"C:\Users\ejmei\repos\ModalAnalysis.jl")
Pkg.instantiate()
Pkg.resolve()

using ModalAnalysis
using Unitful
using ForceConstants
using StatsBase
using DelimitedFiles

#TODO: get it to handle units
#TODO: write data in another CPU thread??

###############
# NMA EXAMPLE #
###############
# base_path = raw"C:\Users\ejmei\repos\ModalAnalysis.jl\examples\LJ_FCC_4UC\NMA"
# base_path = raw"C:\Users\ejmei\Desktop\seed4"
base_path = raw"C:\Users\ejmei\Desktop\WTF\WorkedB4"
equilibrium_data_path = joinpath(base_path, "equilibrium.atom")
dump_path = joinpath(base_path, "dump.atom")
thermo_path = joinpath(base_path,"thermo_data.txt")

eq = LammpsDump(equilibrium_data_path);
parse_timestep!(eq, 1)
atom_masses = get_col(eq, "mass")

ld = LammpsDump(dump_path);

#Load Thermo Data & Masses
pe_col = 3; T_col = 2
thermo_data = readdlm(thermo_path, skipstart = 2);
potential_eng_MD = thermo_data[:,pe_col]
temps = thermo_data[:, T_col]

#Replicate potential used in LAMMSP
pot = LJ(3.4, 0.24037, 8.5);

out_path = base_path

T_des = 10.0
T_avg = mean(temps)

#Dump needs xu,yu,zu
gpu_id = 0
TEP_path = raw"C:\Users\ejmei\repos\ModalAnalysis.jl\examples\LJ_FCC_4UC\K3_4UC_LJ.jld2"
kB = ustrip(u"kcal * mol^-1 * K^-1", Unitful.k*Unitful.Na)
NMA(eq, ld, potential_eng_MD, atom_masses, out_path,  TEP_path, gpu_id)
NM_postprocess(base_path, dirname(TEP_path), kB, T_avg; average_identical_freqs = true)
# ProfileView.@profview


# NMA(eq, ld, pot, potential_eng_MD, atom_masses, out_path, gpu_id)
###############
# INMA EXAMPLE #
###############
# max_deviation = 0.01*mean(potential_eng_MD)

#Dump needs x, y, z, ix, iy, iz, fx, fy, fz
# INMA(ld, pot, potential_eng_MD, masses, max_deviation, out_path, T_des, kB)   