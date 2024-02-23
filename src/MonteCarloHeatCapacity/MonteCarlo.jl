export MC_Simulation, runMC, pick_step_size

struct MC_Simulation{S,L,T,B}
    n_steps::Int
    n_steps_equilibrate::Int
    sampling_dist::Normal
    initial_posns::Vector{Vector{L}}
    temp::T
    beta::B
end

function MC_Simulation(n_steps::Int, n_steps_equilibrate::Int, step_size_std, temp, kB, initial_posns)
    beta = 1/(kB*temp)
    sampling_dist =  Normal(0,ustrip(step_size_std))
    return MC_Simulation{typeof(step_size_std), eltype(initial_posns[1]), typeof(temp), typeof(beta)}(
        n_steps, n_steps_equilibrate, sampling_dist, initial_posns, temp, beta)
end

mutable struct PositionStorage{L,U}
    const r::Vector{Vector{L}}
    const r_uw::Matrix{L}
    disp::Vector{U}
    disp_new::Vector{U}
    r_uw_out::Array{L,3}
end

save_data(ps::PositionStorage, outpath, idx_rng, ::Val{:None}) = nothing

function save_data(ps::PositionStorage, outpath, idx_rng::StepRange, ::Val{:NMA})
    jldopen(joinpath(outpath, "mc_unwrapped_coords.jld2"), "a+") do file
        for k in range(1, length(idx_rng))
            file["r_uw$(idx_rng[k])"] = DataFrame(xu = view(ps.r_uw_out, :, 1, k),
                                                  yu = view(ps.r_uw_out, :, 2, k),
                                                  zu = view(ps.r_uw_out, :, 3, k))
        end
    end
end

function save_data(ps::PositionStorage, outpath, idx, ::Val{:INMA})
    error("Not implemented yet")
end

function calculate_save_interlval(N_atoms, D, T, allowed_mem_GB = 0.5)
    nbytes_per_step = N_atoms*D*sizeof(T)
    nGB_per_step = nbytes_per_step/(1024^3)
    return floor(Int,allowed_mem_GB/nGB_per_step)
end

function (sim::MC_Simulation)(sys::SuperCellSystem{D}, F2::Array{T,2}, F3::Array{T,3},
     U_current, disp_idxs, ps::PositionStorage) where {D,T}
    
    accepted = false

    #Pick a random particle to perturb
    i = sample(1:n_atoms(sys))

    #1D index of displacements
    disp_idxs[1] = D*(i - 1) + 1
    disp_idxs[2] = D*(i - 1) + 2
    disp_idxs[3] = D*(i - 1) + 3

    #Generate random pertubation in x,y,z
    Δr = rand(sim.sampling_dist,3)

    #Update position
    ps.r[i] .+= Δr
    ps.r_uw[i,:] .+= Δr
    ps.disp_new[disp_idxs] .+= Δr #* wrong?

    #Enforce PBC
    ps.r[i] = enforce_cell_size!(ps.r[i], ustrip.(sys.box_sizes_SC))

    #Get energy contribution of atom j at its new position
    delta_U = ΔU_TEP_atomic(F2, F3, disp_idxs..., ps.disp, ps.disp_new)

    if delta_U < 0
        U_current += delta_U
        accepted = true
        ps.disp[disp_idxs] .+= Δr #update prev disp to match this step
    else
        p_accept = exp(-sim.beta*delta_U)
        accept_move = sample([false,true], ProbabilityWeights([1 - p_accept, p_accept]))
        if accept_move
            accepted = true
            U_current += delta_U
            ps.disp[disp_idxs] .+= Δr #update prev disp to match this step
        else # rejected, revert changes
            ps.r[i] .-= Δr
            ps.r_uw[i,:] .-= Δr
            ps.disp_new[disp_idxs] .-= Δr
        end
    end

    return ps, U_current, accepted
end

function (sim::MC_Simulation)(sys::SuperCellSystem{D}, pot::Potential, U_current, disp_idxs, ps::PositionStorage) where D
    # #TODO copy version that works for LJ from old code
    # normal_dist = Normal(0,ustrip(sim.step_size_std))
        
    # #Pick a random particle to perturb
    # j = sample(1:sys.n_particles) #*always allocates?

    # #Save its position incase peturbation is rejected
    # r_temp = deepcopy(position(sys, j)) #*always allocates
    # r_temp_uw = deepcopy(unwrapped_position(sys,j))

    # #Update position and get new contribution
    # Δr = rand(normal_dist,3)*length_unit(pot)
    # update_position(sys, j, position(sys, j) .+ Δr)
    # update_unwrapped_position(sys, j, unwrapped_position(sys, j) .+ Δr)

    # #Enforce PBC
    # enforceCellSize!(sys)

    # #Get energy contribution of atom j at its new position
    # U_new = ΔU_TEP_atomic(F2, F3, iα, iβ, iγ, disp_last, disp_current)
    # delta_U = U_new - atom_energies[j]

    # if delta_U < 0*zero(pot.ϵ)
    #     U_total += delta_U
    #     atom_energies[j] += delta_U
    #     accepted = true
    # else
    #     p_accept = exp(-beta*delta_U)
    #     accept_move = sample([false,true], ProbabilityWeights([1 - p_accept, p_accept]))
    #     if accept_move
    #         accepted = true
    #         U_total += delta_U
    #         atom_energies[j] += delta_U
    #     else # rejected
    #         update_position(sys, j, r_temp)
    #         update_unwrapped_position(sys, j, r_temp_uw)
    #     end
    # end

    # return sys, U_total, atom_energies, accepted, j
end

#Generate configurations, energies are calculate with atom-based TEP
function runMC(sys::SuperCellSystem{D}, sim::MC_Simulation,
     outpath, output_type::Symbol, data_interval::Int, F2, F3) where D

    N_atoms = n_atoms(sys)

    #Create copies of positions that I can modify freely
    r = ustrip.(deepcopy(ustrip.(positions(sys))))
    r_uw = permutedims(reshape(deepcopy(reduce(vcat,r)), (D, N_atoms)), (2,1))
    u = zeros(D*N_atoms)

    save_size = calculate_save_interlval(N_atoms, D, typeof(r_uw[1,1]))
    ps = PositionStorage(r, r_uw, u, deepcopy(u), zeros(size(r_uw)..., save_size))

    disp_idxs = zeros(Int64, D)

    #Equilibrate System
    U_current = 0.0
    for _ in range(1,sim.n_steps_equilibrate)
        ps, U_current, _ = sim(sys, F2, F3, U_current, disp_idxs, ps)
    end
    # @info "Equilibration complete"

    #Set values for sampling run
    num_accepted = 0
    U_arr = zeros(sim.n_steps)
    U_arr[1] = U_current

    #Storage to avoid writing every step
    ps.r_uw_out[:,:,1] .= ps.r_uw
    current_save_idx = 2
    last_save_idx = 0
    n_traj_saved = 1

    for idx in range(2,sim.n_steps)
        ps, U_arr[idx], accepted = sim(sys, F2, F3, U_arr[idx-1], disp_idxs, ps)
        num_accepted += accepted


        if (idx-1) % data_interval == 0
            ps.r_uw_out[:,:,current_save_idx] .= ps.r_uw

            if current_save_idx == save_size
                n_traj_saved += length((last_save_idx+1):data_interval:idx)
                save_data(ps, outpath, (last_save_idx+1):data_interval:idx, Val(output_type))
                current_save_idx = 1
                last_save_idx = idx
            end

            current_save_idx += 1
        end

    end

    #Save whatever is currently in the buffer
    remaining_saves = (last_save_idx+1):data_interval:sim.n_steps
    n_traj_saved += length(remaining_saves)
    save_data(ps, outpath, remaining_saves, Val(output_type))

    jldopen(joinpath(outpath, "mc_unwrapped_coords.jld2"), "a+") do file
        file["N_trajectories"] = n_traj_saved
        file["Interval"] = data_interval
    end

    return U_arr, num_accepted
end

#Crude but at least its automatic
# Grid search valaues based on provided length scale
function pick_step_size(sys::SuperCellSystem{D}, n_steps, length_scale, F2, F3, temp, kB) where D

    max_accepted = 70
    min_accepted = 30

    percent_accepted = 0.0
    step_size = 0.005*length_scale #start with small dx and increase from there
    dx = 0.01*length_scale

    #Increase step size until at most 70% accepted
    while true
        percent_accepted = step_size_iter(sys, n_steps, F2, F3, temp, kB, step_size)
        # @info "$(step_size) $(percent_accepted) $(max_accepted)"
        if percent_accepted < max_accepted
            break
        else
            step_size += dx
        end
    end

    if percent_accepted < min_accepted
        @warn "Automatic step size selection might have failed. Ended with $(percent_accepted)% accepted
                    and step size $(step_size)"
    end
    
    return step_size, percent_accepted
end

function step_size_iter(sys::SuperCellSystem{D}, n_steps, F2, F3, temp, kB, step_size) where D

    U_current = 0.0
    disp_idxs = zeros(Int64, D)
    N_atoms = n_atoms(sys)

    r = ustrip.(deepcopy(ustrip.(positions(sys))))
    r_uw = permutedims(reshape(deepcopy(reduce(vcat,r)), (D, N_atoms)), (2,1))
    u = zeros(D*N_atoms)

    save_size = 1
    ps = PositionStorage(r, r_uw, u, deepcopy(u), zeros(size(r_uw)..., save_size))
    n_accept = 0

    sim = MC_Simulation(n_steps, n_steps, step_size, temp, kB, deepcopy(positions(sys)))

    for _ in range(1,n_steps)
        _, _, accepted = sim(sys, F2, F3, U_current, disp_idxs, ps)
        n_accept += accepted
    end

    return 100*n_accept/n_steps

end

#Energy change when single atom is moved
function U_single(r, pot::PairPotential, box_sizes, j::Int)
    U = zero(pot.ϵ)

    r_ij = zeros(length(box_sizes))*length_unit(pot)
    r_cut_sq = pot.r_cut*pot.r_cut
    for i in range(1,n_particles(sys))           
        if i != j
            #Vector between particle i and j
            r_ij .= r[i] .- r[j]

            nearest_mirror!(r_ij, box_sizes)
            dist_sq = dot(r_ij, r_ij)
            
            #Make sure mirrored particle is in cuttoff
            if dist_sq < r_cut_sq
                U += potential(pot, sqrt(dist_sq))
            end
        end
    end
    
    return U
end

function U_single(r, pot::StillingerWeberSilicon, box_sizes, j::Int)
    #TODO
end

### Functions to Enfroce PBC ###

function nearest_mirror!(r_ij, box_sizes)
    for i in eachindex(box_sizes)  
        if r_ij[i] > box_sizes[i]/2
            r_ij[i] -= box_sizes[i]
        elseif r_ij[i] < -box_sizes[i]/2
            r_ij[i] += box_sizes[i]
        end
    end
    
    return r_ij
end
                                             

function enforce_cell_size!(r, box_sizes)

    if r[1] < zero(r[1]) || r[1] > box_sizes[1]
        r[1] = r[1] - sign(r[1])*box_sizes[1]
    elseif  r[2] < zero(r[2]) || r[2] > box_sizes[2]
        r[2] = r[2] - sign(r[2])*box_sizes[2]
    elseif  r[3] < zero(r[3]) || r[3] > box_sizes[3]
        r[3] = r[3] - sign(r[3])*box_sizes[3]
    end

   return r

end