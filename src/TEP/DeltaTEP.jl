export ΔU_TEP_atomic


"""
This file contains functions to calcualte change in TEP energy when only 
one element of u changes at a time (MonteCarlo). Currently up to third order 
is implemented.
"""

# Takes advantage of sparsity in Δu, but not sparsity in F2
# # i, j, k are indices of elements in u that changed
function ΔU2(F2, u, i, j, k, Δuᵢ, Δuⱼ, Δuₖ)

    ΔU2 = 0.0
    
    # Col w/ just one pertubation affecting, ignore row: taylor series double count (e.g. div by 2)
    # This will overlap with the edge cases below, do not double count Δu*u terms
    ΔU2 += @turbo @views Δuᵢ * (u' * F2[:,i]) + Δuⱼ * (u' * F2[:,j]) + Δuₖ * (u' * F2[:,k])
    # ΔU2 += sum( @views @. u * ((Δuᵢ * F2[:,i]) + (Δuⱼ * F2[:,j]) + (Δuₖ  * F2[:,k])))

    #Row and Cols w/ two pertubations affecting: i,j & i,k & j,k
    ΔU2 += (Δuᵢ*Δuⱼ * F2[i,j])
    ΔU2 += (Δuᵢ*Δuₖ * F2[i,k])
    ΔU2 += (Δuₖ*Δuⱼ * F2[j,k])

    # Need 0.5 because the self terms are not double counted in
    # the original Taylor series like the others
    ΔU2 += 0.5 * (Δuᵢ^2) * F2[i,i]
    ΔU2 += 0.5 * (Δuⱼ^2) * F2[j,j]
    ΔU2 += 0.5 * (Δuₖ^2) * F2[k,k]

    return ΔU2
end

function U2_modal(ω, q)
    return 0.5*sum((ω .*q) .^ 2)
end

# Still does unecessary work but less than full matrix multiply
function ΔU3(F3, u_last, u_current, Δu, i, j, k)

    ΔU3_vals = zeros(length(u_last))
    for m in eachindex(u_last) #* could parallelize but specify number of threads
        ΔU3_vals[m] = ΔU3_helper(view(F3,:,:,m), m, u_last, u_current, i, j, k, Δu)
    end

    return sum(ΔU3_vals)
end


#Does not take advantage of sparsity in F3 or Δu
function ΔU3_helper(F3_slice, slice_idx, u, u_current, i, j, k, Δu)

    ΔU3 = 0.0

    if slice_idx ∉ [i,j,k] 
        #only col/row i,j,k change in this slice
        for col in [i,j,k]
            @turbo for row in eachindex(u)
                ΔU3 += (1/3) * F3_slice[row, col] * 
                    (((u_current[col]) * (u_current[row]) * (u_current[slice_idx]))
                    - u[col]*u[row]*u[slice_idx])
            end
        end

        # #Correct for cross terms being double counted
        ΔU3 -= (1/3) * u[slice_idx] * F3_slice[i, j] * (Δu[i]*u[j] + Δu[j]*u[i] + Δu[i]*Δu[j])
        ΔU3 -= (1/3) * u[slice_idx] * F3_slice[i, k] * (Δu[i]*u[k] + Δu[k]*u[i] + Δu[i]*Δu[k])
        ΔU3 -= (1/3) * u[slice_idx] * F3_slice[j, k] * (Δu[j]*u[k] + Δu[k]*u[j] + Δu[j]*Δu[k])

        # Correct for self terms being over-counted
        ΔU3 -= (1/6) * F3_slice[i, i] * (u[slice_idx] * (2*Δu[i]*u[i] + Δu[i]*Δu[i]))
        ΔU3 -= (1/6) * F3_slice[j, j] * (u[slice_idx] * (2*Δu[j]*u[j] + Δu[j]*Δu[j]))
        ΔU3 -= (1/6) * F3_slice[k, k] * (u[slice_idx] * (2*Δu[k]*u[k] + Δu[k]*Δu[k]))

    else
        #All terms in this slice change -- loop terms below diagonal to avoid double counting
        for col in eachindex(u)
            @turbo for row in range(col,length(u))
                # Some of these F3 are zero, and most of the Δu are zero as well --> wasted time
                ΔU3 += (1/3) * F3_slice[row, col] * 
                    (((u_current[col]) * (u_current[row]) * (u_current[slice_idx]))
                    - u[col]*u[row]*u[slice_idx])
            end
        end
        # Correct for diagonal terms being over-counted
        @turbo for i in range(1,length(u))
            ΔU3 -= (1/6) * F3_slice[i, i] * (u[slice_idx] * (2*Δu[i]*u[i] + Δu[i]*Δu[i]) + Δu[slice_idx]* (u[i]*u[i] + 2*Δu[i]*u[i] + Δu[i]*Δu[i]))
        end
    end

    return ΔU3
end


function ΔU_TEP_atomic(F2::Array{T,2}, F3::Array{T,3}, iα, iβ, iγ, u_last, u_current) where T
    Δu = u_current .- u_last #& allocation
    return ΔU2(F2, u_last, iα, iβ, iγ, Δu[[iα,iβ,iγ]]...) + ΔU3(F3, u_last, u_current, Δu, iα, iβ, iγ)
end

function ΔU_TEP_atomic(F2::Array{T,3}, i, j, k, u_last, Δuᵢ, Δuⱼ, Δuₖ) where T
    return ΔU2(F2, u_last, i, j, k, Δuᵢ, Δuⱼ, Δuₖ)
end


#Not sure this works but its easier to understand that what I did above
# function ΔU_TEP_atomic2_test(F2, i, u_current, u_last, U_TEP2_last)
#     ΔU = 0.0
#     D = length(Δui)
#     N_atoms = Int64(length(u_last) / D)
#     for j in range(1,N_atoms)
#         if i ≠ j
#             for α in range(1,D)
#                 ii = 3*(i-1) + α
#                 for β in range(1,D)
#                     jj = 3*(j-1) + β
#                     ΔU += (F2[ii,jj] * u_current[ii] * u_last[jj])
#                 end
#             end
#         end
#     end

#     #Do self block
#     for α in range(1,D)
#         ii = 3*(i-1) + α
#         for β in range(1,D)
#             jj = 3*(j-1) + β
#             ΔU += (F2[ii,jj] * u_current[i] * u_current[j])
#         end
#     end

#     return (ΔU/2) - U_TEP2_last[ii]
# end

# # # #Test delta multiplication
# function sym_2D(N)
#     unique_idx = with_replacement_combinations(range(1,N), 2)
#     arr = zeros(N,N)
#     for idx in unique_idx
#         non_unique_idx = permutations(idx)
#         a = 2*rand(Float64) - 1
#         for n_idx in non_unique_idx
#             arr[n_idx...] = a
#         end
#     end
#     return arr
# end

# function sym_3D(N)
#     unique_idx = with_replacement_combinations(range(1,N), 3)
#     arr = zeros(N,N,N)
#     for idx in unique_idx
#         non_unique_idx = permutations(idx)
#         a = 2*rand(Float64) - 1
#         for n_idx in non_unique_idx
#             arr[n_idx...] = a
#         end
#     end
#     return arr
# end
