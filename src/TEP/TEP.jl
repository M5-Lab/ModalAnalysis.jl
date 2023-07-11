function U_TEP3_n(F3, u, n)
    F3_view = view(F3, :, :, n)
    u_n = u[n]
    @tensor begin
        U3 = u_n*F3_view[j,k]*u[j]*u[k]
    end
    return U3/6
end


# function U_TEP3_n(F3::ThirdOrderSparse)

# end






## Brute Force Versions for Testing ##

function U_TEP_bf(F2, F3, u)    
    #Calculate potential energy
    U2 = 0.5*((transpose(u) * F2) * u) # ~560 μs with 256 atoms
    U3 = zero(U2)

    @turbo for k in eachindex(u)
        for j in eachindex(u)
            for i in eachindex(u)
                U3 += F3[i, j, k] * (u[i]*u[j]*u[k])
            end
        end
    end


    return U2 + U3/6
end

function U_TEP3_n_bf(F3,u,n)

    U3 = 0.0
    u_n = u[n]

    @turbo for j in eachindex(u)
        for i in eachindex(u)
            U3 += F3[i, j, n] * (u[i]*u[j]*u_n)
        end
    end
    return U3/6
end