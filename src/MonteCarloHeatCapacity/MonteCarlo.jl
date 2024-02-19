struct MC_Simulation{S,T}
    n_steps::Int
    n_steps_equilibrate::Int
    step_size_std::S
    temp::T
end

#Generate configurations, energies are calculate with atom-based TEP
function runMC!(fc_data::ForceConstantData, dynmat_data::DynamicalMatrix,
    sys::System{D}, sim::MC_Simulation, pot::Potential) where D

    beta = 1/(sys.kB*sim.temp)

    #Equilibrate System
    U_current = zero(pot.ϵ)
    for i in range(1,sim.n_steps_equilibrate)
        sys, U_current, _, _ = monte_carlo_step!(sys, U_current, beta, pot, sim)
    end
    @info "Equilibration complete"

    num_accepted = 0
    U_arr = zeros(sim.n_steps)*energy_unit(pot)
    U_TEP_atomic2 = zeros(sim.n_steps)*energy_unit(pot)
    U_TEP_atomic3 = zeros(sim.n_steps)*energy_unit(pot)

    #Set Initial Values
    U_arr[1] = U_current
    U_TEP_atomic2[1] = U_current
    U_TEP_atomic3[1] = U_current

    #Initial displacements
    u_last = zeros(D*sys.n_particles)*length_unit(pot)
    disp_idxs = zeros(D)*length_unit(pot)
    for idx in range(2,sim.n_steps)

        #Update positions with MonteCarlo
        sys, U_current, accepted, particle_idx = monte_carlo_step!(sys, U_current, beta, pot, sim)
        num_accepted += accepted

        U_arr[idx] = U_current

        if accepted
            #1D index of displacements
            disp_idxs[1] = D*(particle_idx - 1) + 1
            disp_idxs[2] = D*(particle_idx - 1) + 2
            disp_idxs[3] = D*(particle_idx - 1) + 3

            #Calculate energy from TEP in real space
            u_current = displacement_1D(sys) #uses unwrapped coordinates
            Δu = u_current[disp_idxs] .- u_last[disp_idxs]

            ΔU_atomic2 = ΔU_TEP_atomic2(fc_data.F2, disp_idxs..., ustrip.(u_last), ustrip.(Δu)...) * energy_unit(pot) 
            ΔU_atomic3 = ΔU_TEP_atomic3(fc_data.F2, fc_data.F3, disp_idxs...,
                    ustrip.(u_last), ustrip.(u_current)) * energy_unit(pot)

            U_TEP_atomic2[idx] = U_TEP_atomic2[idx-1] + ΔU_atomic2
            U_TEP_atomic3[idx] = U_TEP_atomic3[idx-1] + ΔU_atomic3

            u_last = u_current
        else
            U_TEP_atomic2[idx] = U_TEP_atomic2[idx-1]
            U_TEP_atomic3[idx] = U_TEP_atomic3[idx-1]
        end
    end

    return U_arr, U_TEP_atomic2, U_TEP_atomic3, U_TEP_modal, num_accepted
end

#Makes use of global variable kB
function monte_carlo_step!(sys::System{D}, U_total, atom_energies, beta, sim::MC_Simulation) where D
    accepted = false

    #Distribution of possible step sizes
    normal_dist = Normal(0,ustrip(sim.step_size_std))

    #Pick a random particle to perturb
    j = sample(1:sys.n_particles) #*always allocates?

    #Save its position incase peturbation is rejected
    r_temp = deepcopy(position(sys, j)) #*always allocates
    r_temp_uw = deepcopy(unwrapped_position(sys,j))

    #Update position and get new contribution
    Δr = rand(normal_dist,3)*length_unit(pot)
    update_position(sys, j, position(sys, j) .+ Δr)
    update_unwrapped_position(sys, j, unwrapped_position(sys, j) .+ Δr)

    #Enforce PBC
    enforceCellSize!(sys)

    #Get energy contribution of atom j at its new position
    U_new = ΔU_TEP_atomic(F2, F3, iα, iβ, iγ, disp_last, disp_current)
    delta_U = U_new - atom_energies[j]

    if delta_U < 0*zero(pot.ϵ)
        U_total += delta_U
        atom_energies[j] += delta_U
        accepted = true
    else
        p_accept = exp(-beta*delta_U)
        accept_move = sample([false,true], ProbabilityWeights([1 - p_accept, p_accept]))
        if accept_move
            accepted = true
            U_total += delta_U
            atom_energies[j] += delta_U
        else # rejected
            update_position(sys, j, r_temp)
            update_unwrapped_position(sys, j, r_temp_uw)
        end
    end

    return sys, U_total, atom_energies, accepted, j
end


### Functions to Enfroce PBC ###

function nearest_mirror(r_ij,L)
    r_x = r_ij[1]; r_y = r_ij[2]; r_z = r_ij[3]

    if r_x > L/2
        r_x = r_x - L
    elseif r_x < -L/2
        r_x = r_x + L
    end

    if r_y > L/2
        r_y = r_y - L
    elseif r_y < -L/2
        r_y = r_y + L  
    end

    if r_z > L/2
        r_z = r_z - L
    elseif r_z < -L/2
        r_z = r_z + L
    end

    return [r_x,r_y,r_z] 
end
                                             
function enforceCellSize!(sys::System)

    L = sys.box_size
    for i in eachindex(sys)
        r_x, r_y, r_z = position(sys,i)

        if r_x < zero(r_x) || r_x > L
            r_x = r_x - sign(r_x)*L
        elseif  r_y < zero(r_y) || r_y > L
            r_y = r_y - sign(r_y)*L
        elseif  r_z < zero(r_z) || r_z > L
            r_z = r_z - sign(r_z)*L
        end

        update_position(sys, i, [r_x, r_y, r_z])
        # sys.coords[i,:] = [r_x,r_y,r_z]
    end

end