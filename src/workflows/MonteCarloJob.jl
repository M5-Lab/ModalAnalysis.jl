export MonteCarloJob


"""
Assumes positions given in the system are the reference positions to calculate displacements from. 

Generates output that is compatible with `NMA`` or `INMA`.
"""
function MonteCarloJob(sys::SuperCellSystem{D}, TEP_path::String,
     n_steps::Int, n_steps_equil::Int, step_size_std, outpath::String, temp, kB;
     output_type = :NMA, F2_name::String = "F2", F3_name::String = "F3") where D

    if output_type ∉ (:NMA, :INMA)
        error("output_type must be :NMA or :INMA got $(output_type)")
    end


    sim = MC_Simulation(n_steps, n_steps_equil, step_size_std, temp, kB, deepcopy(positions(sys)))

    U_arr, num_accepted = runMC(sys, TEP_path, sim, outpath, output_type; F2_name = F2_name, F3_name = F3_name)

    jldsave(joinpath(outpath, "MC_stats.jld2"), U_arr = U_arr, percent_accepted = 100*num_accepted/sim.n_steps)

end


