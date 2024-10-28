## Canonical Configuration Monte Carlo
using Pkg
Pkg.activate("C:/Users/ejmei/.julia/environments/research")

using Molly
using Unitful
using ForceConstants

temperature = 70u"K"

crys = FCC(5.2468u"Å", 39.95u"g/mol", SVector(4,4,4))
sys = SuperCellSystem(crys);
pot = LJ(3.4u"Å", 0.24037u"kcal/mol", 8.5u"Å")
calc = AnalyticalCalculator(1e-8, pot.r_cut)

ifc_path = raw"Z:/emeitz/Data/ForceConstants/AvgINM_LJ/AvgIFC_LJ_$(ustrip(temperature))K_CLEANED.jld2"
freqs_sq, phi = load(ifc_path, "freqs_sq", "phi");
freqs = sqrt.(freqs_sq);

configs, energies = GenerateConfigs(sys, temps, freqs, phi, n_configs, energy_unit(pot);
                             mode = :classical, pot = pot);


pairwise_inters = (LennardJones(
    cutoff = ShiftedForceCutoff(8.5u"Å"),
    energy_units=u"kJ * mol^-1",
    force_units=u"kJ * mol^-1 * nm^-1",
    use_neighbors = true
),)

n_configs = 50_000


function random_config!(sys::System{D, G, T};
    shift_size = oneunit(eltype(eltype(sys.coords)))) where {D, G, T}

    rand_idx = rand(eachindex(sys))
    direction = random_unit_vector(T, D)
    magnitude = rand(T) * shift_size
    sys.coords[rand_idx] = wrap_coords(sys.coords[rand_idx] .+ (magnitude * direction), sys.boundary)
    return sys
end

sys = System(
    crys,
    pairwise_inters=pairwise_inters,
    loggers=(
        coords=CoordinateLogger(n_atoms, dims=n_dimensions(boundary)),
        montecarlo=MonteCarloLogger(),
    ),
    energy_units=u"kJ * mol^-1",
    force_units=u"kJ * mol^-1 * nm^-1",
)

# Update sigma and epsilon params for each atom
for (i, atom) in enumerate(sys.atoms)
    atom.σ = 3.4u"Å"
    atom.ϵ = 0.24037u"kJ/mol"
end


trial_args = Dict(:shift_size => 0.1u"nm")
sim = MetropolisMonteCarlo(; 
    temperature=t,
    trial_moves=random_uniform_translation!,
    trial_args=trial_args,
)

simulate!(sys, sim, 10_000)

