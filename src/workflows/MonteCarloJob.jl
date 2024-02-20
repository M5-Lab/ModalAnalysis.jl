export MonteCarloJob


"""
Assumes positions given in the system are the reference positions to calculate displacements from. 

Generates output that is compatible with `NMA`` or `INMA`.
"""
function MonteCarloJob(sys::SuperCellSystem{D}, TEP_folder::String,
     n_steps::Int, n_steps_equil::Int, step_size_std, outpath::String;
     output_type = :NMA, F2_name::String = "F2", F3_name::String = "F3")

    if output_type ∉ (:NMA, :INMA)
        error("output_type must be :NMA or :INMA got $(output_type)")
    end

    if energy_unit(pot) == u"eV"
        kB = ustrip(u"eV/K", Unitful.k)
    elseif energy_unit(pot) == u"kcal/mol"
        kB = ustrip(u"kcal * mol^-1 * K^-1", Unitful.k*Unitful.Na)
    else
        error("Unknown unit system")
    end

    sim = MC_Simulation(n_steps, n_steps_equil, step_size_std, deepcopy(positions(sys)), temp, kB)

    U_arr, num_accepted = runMC(sys, TEP_folder, sim, outpath, output_type; F2_name = F2_name, F3_name = F3_name)

    jldsave(joinpath(outpath) U_arr = U_arr, percent_accepted = 100*num_accepted/sim.n_steps)

end


