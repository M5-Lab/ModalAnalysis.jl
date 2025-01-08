# ModalAnalysis.jl
 

This package was built to take output from [ForceConstants.jl](https://github.com/ejmeitz/ForceConstants.jl/tree/main) and LAMMPS simulations to calculate anharmonic phonon heat capacities as described by the paper: "Phonon Mode-Resolved Anharmonic Heat Capacity of Solids".

This package requires access to an NVIDIA GPU to work.

This package is not in the general registry so to install use the command below. Please change the revs to whatever the latest tag is to get up-to-date code!

```julia
using Pkg
Pkg.add(url = "https://github.com/ejmeitz/ForceConstants.jl.git", rev = "v0.1.3")
Pkg.add(url = "https://github.com/M5-Lab/ModalAnalysis.jl.git", rev = "v0.0.1")
```

### Generating IFCs with [ForceConstants.jl](https://github.com/ejmeitz/ForceConstants.jl) 
The code below takes the derviative w.r.t. the equilibrium positions defined by `fcc_crystal`. So these are the Zero-Kevlin force constants. However, they are calculated exactly with automatic differentiation instead of the traditional finite differences.

The important thing to note is the format of the output. If you want to use external force constants, `ModalAnalysis.jl` expects them as a [JLD2](https://github.com/JuliaIO/JLD2.jl) (similar to HDF5) file with the following keys: `freqs_sq`, `dynmat`, `phi`, `K3`. All entries are dense.

```julia
using ForceConstants
using JLD2
using CUDA

pot_lj = LJ(3.4u"Å", 0.24037u"kcal * mol^-1", 8.5u"Å")
fcc_crystal = FCC(5.2468u"Å", :Ar, SVector(4,4,4)) #from SimpleCrystals.jl
sys_lj = SuperCellSystem(fcc_crystal);

# Choose force constant calculator
tol = 1e-12
calc_analytical_lj = AnalyticalCalculator(tol, pot_lj.r_cut)

# Calculate dynamical matrix, mode shapes, frequencies, and third-order IFCs
dynmat = dynamical_matrix(sys_lj, pot_lj, calc_analytical_lj)
freqs_sq, phi = get_modes(dynmat)
ifc3_analytical = third_order(sys_lj, pot_lj, calc_analytical_lj)

# Modifies values in ifc3_analytical
mass_weight_third_order!(ifc3_analytical, ustrip.(masses(sys_lj)))

cuPhi = CuArray{Float32}(phi) # eigenvectors/mode shapes
cuPsi_mw = CuArray{Float32}(ifc3_analytical.values)

K3 = mcc3(cuPsi_mw, cuPhi, 256) # there are 768 DoF, so 256 chosen to make calculation smaller

# These are the inputs required for modal analysis
jldsave("LJ_0K_IFCs.jld2", dynmat = dynmat, phi = phi, freqs_sq = freqs_sq, K3 = K3)
```

### Run LAMMPS to generate data
Inside the `scripts` folder are two LAMMPS input files. One for SW silicon (`NMA_Langevin_SW_3UC.in`) and another for LJ argon (`NMA_Langevin_LJ.in`). These scripts will produce the output format expected by `ModalAnalysis.jl` for a single seed. `ModalAnalysis.jl` expects the output from each seed to be in its own folder (defined by the `sim_folder_name` function below) so to get better statistics you will need to write a script that can change the seed in the LAMMPS script and generate new output in an organized file structure. The file structure defined by the first example below is:

- simulation_folder
     - T100
          - seed0
          - seed 1
          ....
     - T1300
          - seed0
          - seed 1
          ....

### Script to Calculate Anharmonic Mode Heat Capacities
```julia
using ModalAnalysis
using Unitful
import ForceConstants: StillingerWeberSilicon, LJ

sim_folder = "/mnt/merged/emeitz/SW_1300K_MIXED_IFC_TEST"
n_seeds = 50
order = 3

pot_sw = StillingerWeberSilicon()
#Just need energy units so the code and figure out kB
#pot_lj = LJ(3.4, 0.24037u"kcal/mol", 8.5)

temperatures = [100, 1300]

# This path is joined to sim_folder to run the analysis.
# This code expects the output from the LAMMPS scripts
# contained in this folder : equilibrium.energies, equilibrium.atom, thermo_data.txt
sim_folder_name(temp, seed) = ["T$(temp)", "seed$(seed)"]

# Folder containing IFCs in JLD2 format
# Expects dynmat, freqs_sq,phi,  K3 as keys (see readme for how to generate this file)
tep_folder = "/mnt/mntsdb/emeitz/ForceConstants/SW_ALM"
tep_file_name(temp) = "SW_$(temp)K_mixed.jld2"
 
NMA_GPU_Jobs(sim_folder, tep_folder, temperatures,
     sim_folder_name, tep_file_name, n_seeds, pot_sw; order = order)


#########################################################
### Sweep Over Tempearture and Another Parameter(s) ###
#########################################################

other_params_to_sweep = Dict("dt_percents" => [0,1,2,3,4])

# temp must ALWAYS be first and seed must ALWAYS be last arg, other parmaters passed in between
sim_folder_name(temp, dt_percent, seed) = ["T$(temp)_dT$(dt_percent)", "seed$(seed)"]
# Extra params passed after temperature
tep_file_name(temp, p) = "SW_$(temp)K_$(p).jld2"


NMA_GPU_Jobs(sim_folder, tep_folder, temperatures,
     sim_folder_name, tep_file_name, n_seeds, pot_sw;
     order = order, other_params_to_sweep = other_params_to_sweep)

```

### Script to Calculate AvgIFCs
Inside the `scripts` folder are two LAMMPS input files. One for SW silicon (`AvgINM_SW_3UC.in`) and another for LJ argon (`AvgINM_LJ.in`). Note that these scripts use the Nose-Hoover instead of Langevin. Running these scripts will produce the output required to calculate a set of AvgIFCs. The code below shows how to call the workflow in `ModalAnalysis.jl`

```julia
using ModalAnalysis
using Unitful
import ForceConstants: StillingerWeberSilicon, LJ, AutoDiffCalculator, AnalyticalCalculator

pot = LJ(3.4, 0.24037u"kcal * mol^-1", 8.5)
calc_AD = AnalyticalCalculator(1f-8, ustrip(pot.r_cut))

# There's a bug in the AD library that makes compiling the SW code very slow
# pot = StillingerWeberSilicon(units = false, T = Float32)
# calc_AD = AutoDiffCalculator(1f-8, ustrip(pot.r_cut))

temperatures = [10, 80]

sim_base_path = "<path-with-output-from-AvgINM_LJ.in>"

# This function defines a postfix that will be appended to sim_base_base. If you 
# only ran one tempearture this can just be the emptry string.
sim_folder_name = (T) -> "T$(T)"

# Used to define the name of the output as a function of the temperature
filename = (T) -> "AvgIFC_SW_$(T)K"

out_path = sim_base_path
N_atoms = 216

verbose = true

# The code will save IFC values intermitently. This can be
# useful for checking convergence w.r.t the number of configurations.
ncheckpoints = 6 

# This will calculate AvgINMs for each tempearture.
AvgINM_Job(pot, calc_AD, temperatures, sim_base_path,
           sim_folder_name, out_path, N_atoms, filename;
           verbose = verbose, ncheckpoints = ncheckpoints)

```
