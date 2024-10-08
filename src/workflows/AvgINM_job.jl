export AvgINM_Job

"""
AvgINM_Job(pot::Potential, calc::ForceConstantCalculator, temperatures::AbstractVector{<:Real},
            sim_folder_name::Function, out_path::String, N_atoms::Integer,
            outfilename::Function; ncores = Threads.nthreads())

This function calculates the averaged instantaneous force constants for a given potential, `pot`,
    using the ForceConstantCalculator, `calc`, over a range of temperatures, `temperatures`. This
    function will parallelize over tempeartures if there is enough RAM. The function will throw and 
    error if it calculates that there is not enough system RAM available. 

# Arguments:
- `pot::Potential`
    Potential used in the simulation
- `calc::ForceConstantCalculator`
    ForceConstantCalculator used to calculate the instantaneous force constants
- `temperatures::AbstractVector{<:Real}`
    List of tempeartures to calculate the averaged force constants at. These same tempeartures must be present
    in the simulation data.
- `sim_base_path::String`
    Path to base folder of the simulation data.
- `sim_folder_name::Function`
    A function that returns a string given a temeprature (`func(temp)`). This string is joined to
    `sim_base_path` to access the data for a given temperature.
- `out_path::String`
    Path to save the averaged force constants.
- `N_atoms::Integer`
    Number of atoms in the system.
- `outfilename::Function
    Function that returns a string given a temperature (`func(temp)`). This string is appended to `out_path`
    to save the averaged force constants. This should not include the file extension.
- `ncores::Integer = Threads.nthreads()`
    Number of CPU cores to use. By default this is `Threads.nthreads()`. The function will
    attempt to parallelize over temperatures if there is enough RAM.
"""
function AvgINM_Job(pot::Potential, calc::ForceConstantCalculator, temperatures::AbstractVector{<:Real},
            sim_base_path::String, sim_folder_name::Function, out_path::String, N_atoms::Integer,
            outfilename::Function; ncores = Threads.nthreads(), verbose = false, ncheckpoints = 1, T = Float32)

    N_IFC3 = (3*N_atoms)^3
    N_bytes_IFC3 = sizeof(T)*N_IFC3
    N_GB_total = 2*N_bytes_IFC3*ncores/(1024^3) #code stores 2 copies of IFC3 in mem
    total_mem_GB = T(Sys.total_memory())/(1024^3)
    @assert total_mem_GB > 1.1*N_GB_total "Not enough memory to use $(ncores) cores.
         $(N_GB_total) GB needed, $(total_mem_GB) GB available."

    n_temps = length(temperatures)

    #* probably shouldnt parallelize this since dynmat, F3 use threads as well
    Threads.@threads for chunk_i in 1:ncores 
        for i in chunk_i:ncores:n_temps
            temp = temperatures[i]
            @info "Starting temperature: $(temp)"
            seed_path = joinpath(sim_base_path, sim_folder_name(temp))

            inma = InstantaneousNormalModeAnalysis(seed_path, pot, temp, calc; req_img_flags = false)

            avg_forces, avg_dynmat, avg_psi = get_average_INMs(inma, calc;
                 verbose = verbose, ncheckpoints = ncheckpoints,
                 filename = outfilename(temp), T = T)

            # freqs_sq, phi = get_modes(avg_dynmat)

            jldsave(joinpath(out_path, outfilename(temp) * "_N$(inma.ld.n_samples)" * ".jld2"), 
                f0 = avg_forces, dynmat = avg_dynmat, 
                F3 = avg_psi, nsamples = inma.ld.n_samples
            )
            @info "Finished temperature: $(temp)"
        end
    end

end