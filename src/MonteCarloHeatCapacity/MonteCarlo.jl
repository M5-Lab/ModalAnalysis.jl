struct MC_Simulation{S,L,T,B}
    n_steps::Int
    n_steps_equilibrate::Int
    sampling_dist::Normal
    initial_posns::Vector{Vector{L}}
    temp::T
    beta::B
end

function MC_Simulation(n_steps::Int, n_steps_equilibrate::Int, step_size_std, temp, kB; initial_posns)
    beta = 1/(kB*temp)
    sampling_dist =  Normal(0,ustrip(sim.step_size_std))
    return MC_Simulation{typeof(step_size_std), eltype(initial_posns[1]), typeof(temp), typeof(beta)}(
        n_steps, n_steps_equilibrate, sampling_dist, initial_posns, temp, beta)
end

struct PositionStorage{L}
    r::Vector{Vector{L}}
    r_uw::Matrix{L}
    disp::Vector{Vector{L}}
    disp_new::Vector{Vector{L}}
end

function save_data(ps::PositionStorage, outpath, idx, ::Val(:NMA))
    jldopen(joinpath(outpath, "mc_unwrapped_coords.jld2"), "a+") do file
        file["r_uw$(idx)"] = DataFrame(xu = view(ps.r_uw, :, 1),
                                       yu = view(ps.r_uw, :, 2),
                                       zu = view(ps.r_uw, :, 3))
    end
end

function save_data(ps::PositionStorage, outpath, idx, ::Val(:INMA))
    error("Not implemented yet")
end

function (sim::MC_Simulation)(sys::SuperCellSystem{D}, F2::Array{T,2}, F3::Array{T,3},
     U_current, disp_idxs, ps::PositionStorage) where {D,T}
    
    accepted = false

    #Pick a random particle to perturb
    i = sample(1:sys.n_particles)

    #1D index of displacements
    disp_idxs[1] = D*(i - 1) + 1
    disp_idxs[2] = D*(i - 1) + 2
    disp_idxs[3] = D*(i - 1) + 3

    #Generate random pertubation in x,y,z
    Δr = rand(sim.sampling_dist,3)*length_unit(pot)

    #Update position
    ps.r[i] .+= Δr
    ps.r_uw[i,:] .+= Δr
    ps.disp_new[i] .+= Δr

    #Enforce PBC
    ps.r[i] = enforce_cell_size!(ps.r[i], sys.box_sizes_SC)

    #Get energy contribution of atom j at its new position
    delta_U = ΔU_TEP_atomic(F2, F3, disp_idxs..., ps.disp, ps.disp_new)

    if delta_U < 0*zero(pot.ϵ)
        U_current += delta_U
        accepted = true
    else
        p_accept = exp(-sim.beta*delta_U)
        accept_move = sample([false,true], ProbabilityWeights([1 - p_accept, p_accept]))
        if accept_move
            accepted = true
            U_current += delta_U
        else # rejected
            ps.r[i] .-= Δr
            ps.r_uw[i,:] .-= Δr
            ps.disp_new[i] .-= Δr
        end
    end

    ps.disp .= ps.disp_new

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
function runMC(sys::SuperCellSystem{D}, TEP_path::String, sim::MC_Simulation, outpath, output_type;
     F2_name::String = "F2", F3_name::String = "F3") where D

    N_atoms = n_atoms(sys)

    F2, F3 = load(TEP_path, F2_name, F3_name)

    energy_unit = 
    length_unit = unit(position(sys,1,1))

    #Create copies of positions that I can modify freely
    r = deepcopy(positions(sys))
    r_uw = permutedims(reshape(deepcopy(r), (D, N_atoms)), (2,1))
    u = zeros(size(r))
    ps = PositionStorage(r, r_uw, u, deepcopy(u))

    disp_idxs = zeros(D)*length_unit

    #Equilibrate System
    U_current = 0.0*energy_unit
    for _ in range(1,sim.n_steps_equilibrate)
        ps, U_current, _ = sim(sys, F2, F3, U_current, disp_idxs, ps)
    end
    @info "Equilibration complete"

    #Set values for sampling run
    num_accepted = 0
    U_arr = zeros(sim.n_steps)*energy_unit
    U_arr[1] = U_current

    save_data(ps, outpath, 1, Val(output_type))

    for idx in range(2,sim.n_steps)
        ps, U_arr[idx], accepted = sim(sys, F2, F3, U_arr[idx-1], disp_idxs, ps)
        num_accepted += accepted

        #* check if this is bottleneck, could write every N steps to disk at cost of memory
        #* Could make separate channel/thread and copy data there. 
        save_data(ps, outpath, idx, Val(output_type))
    end

    return U_arr, num_accepted
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