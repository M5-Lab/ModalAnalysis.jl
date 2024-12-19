# ModalAnalysis.jl
 

This package was built to take output from [ForceConstants.jl](https://github.com/ejmeitz/ForceConstants.jl/tree/main) and LAMMPS simulations to calculate anharmonic phonon heat capacities as described by the paper: "Phonon Mode-Resolved Anharmonic Heat Capacity of Solids".

This package requires access to an NVIDIA GPU to work.

This package is not in the general registry so to install use the command below. Please change rev to whatever the latest tag is to get up-to-date code!

```julia
using Pkg
Pkg.add(url = "https://github.com/M5-Lab/ModalAnalysis.jl.git", rev = "v0.0.1")
```

### Using ForceConstants.jl to generate IFC input:
```julia
#TODO
```

### Run LAMMPS to generate data
#TODO

### Script to Calculate Anharmonic Mode Heat Capacities
```julia
using ModalAnalysis
using Unitful
import ForceConstants: StillingerWeberSilicon, LJ

sim_folder = "/mnt/merged/emeitz/SW_1300K_MIXED_IFC_TEST"
n_seeds = 50
order = 3
temperatures = [100, 1300]


sim_folder_name(temp, seed) = ["T$(temp)", "seed$(seed)"]
tep_folder = "/mnt/mntsdb/emeitz/ForceConstants/SW_ALM" # folder where JLD2 files containing output from ForceConstants.jl are
tep_file_name(temp) = "SW_$(temp)K_mixed.jld2" # if made manually with JLD2 this should have the following keys: dynmat, freqs_sq, phi, K3

#Just need energy units unless you're calculating IFC with the potentials
pot = StillingerWeberSilicon()
#pot = LJ(3.4, 0.24037u"kcal/mol", 8.5)

NMA_GPU_Jobs(sim_folder, tep_folder, temperatures,
     sim_folder_name, tep_file_name, n_seeds, pot; order = order)#, other_params_to_sweep = other_params_to_sweep)

### Sweep other parameters
sim_folder_name(temp, dt_percent, seed) = ["T$(temp)_dT$(dt_percent)", "seed$(seed)"]
tep_file_name(temp, p) = "SW_$(temp)K_$(p).jld2 # this should have the following keys: dynmat, freqs_sq, phi, K3

other_params_to_sweep = Dict("dt_percents" => [0,1,2,3,4])

NMA_GPU_Jobs(sim_folder, tep_folder, temperatures,
     sim_folder_name, tep_file_name, n_seeds, pot; order = order)#, other_params_to_sweep = other_params_to_sweep)

```

