struct MC_Simulation{S,L,T,B}
    n_steps::Int
    n_steps_equilibrate::Int
    step_size_std::S
    reference_positions::Union{Vector{Vector{L}} , Nothing}
    temp::T
    beta::B
end

function MC_Simulation(n_steps::Int, n_steps_equilibrate::Int, step_size_std, temp, kB; reference_positions = nothing)
    beta = 1/(kB*temp)
    if reference_positions !== nothing
        L = eltype(reference_positions[1])
    else
        L = Nothing
    end
    return MC_Simulation{typeof(step_size_std),L , typeof(temp), typeof(beta)}(
        n_steps, n_steps_equilibrate, step_size_std, reference_positions, temp, beta)
end

struct PositionStorage{L}
    r::Vector{Vector{L}}
    r_uw::Vector{Vector{L}}
    disp::Vector{Vector{L}}
    disp_new::Vector{Vector{L}}
end

TEP_Atomic_Types = Union{TEP_Atomic2{CPU_Storage}, TEP_Atomic3{CPU_Storage}}
function (sim::MC_Simulation)(sys::SuperCellSystem{D}, tep::TEP_Atomic_Types, U_current, disp_idxs, ps::PositionStorage) where D
    
    accepted = false

    #Distribution of possible step sizes
    normal_dist = Normal(0,ustrip(sim.step_size_std))

    #Pick a random particle to perturb
    i = sample(1:sys.n_particles)

    #1D index of displacements
    disp_idxs[1] = D*(i - 1) + 1
    disp_idxs[2] = D*(i - 1) + 2
    disp_idxs[3] = D*(i - 1) + 3

    #Generate random pertubation in x,y,z
    Δr = rand(normal_dist,3)*length_unit(pot)

    #Update position
    ps.r[i] .+= Δr
    ps.r_uw[i] .+= Δr
    ps.disp_new[i] .+= Δr

    #Enforce PBC
    ps.r[i] = enforce_cell_size!(ps.r[i], sys.box_sizes_SC)

    #Get energy contribution of atom j at its new position
    delta_U = ΔU_TEP_atomic(tep.F2, tep.F3, disp_idxs..., ps.disp, ps.disp_new)

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
            ps.r_uw[i] .-= Δr
            ps.disp_new[i] .-= Δr
        end
    end

    ps.disp .= ps.disp_new

    return ps, U_current, accepted
end

function save_unwrapped_coords(ps::PositionStorage, outpath, idx)
    jldopen(joinpath(outpath, "mc_unwrapped_coords.jld2"), "a+") do file
        file["r_uw$(idx)"] = ps.r_uw
    end
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
function run(sys::SuperCellSystem{D}, tep::TEP_Atomic_Types, sim::MC_Simulation, outpath) where D

    #Create copies of positions that I can modify freely
    r = deepcopy(positions(sys))
    #* INIT R_ UW
    u = r .- sim.reference_positions
    ps = PositionStorage(r, u, deepcopy(u))

    disp_idxs = zeros(D)*length_unit(tep)

    #Equilibrate System
    U_current = 0.0*energy_unit(tep)
    for _ in range(1,sim.n_steps_equilibrate)
        ps, U_current, _ = sim(sys, tep, U_current, disp_idxs, ps)
    end
    @info "Equilibration complete"

    #Set values for sampling run
    num_accepted = 0
    U_arr = zeros(sim.n_steps)*energy_unit(tep)
    U_arr[1] = U_current

    save_unwrapped_coords(ps, outpath, 1)

    for idx in range(2,sim.n_steps)
        ps, U_arr[idx], accepted = sim(sys, tep, U_arr[idx-1], disp_idxs, ps)
        num_accepted += accepted

        #* check if this is bottleneck, could write every N steps to disk at cost of memory
        #* Could make separate channel/thread and copy data there. 
        save_unwrapped_coords(ps, outpath, idx)
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