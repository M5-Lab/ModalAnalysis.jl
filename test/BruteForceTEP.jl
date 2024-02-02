export U_TEP3_bf, U_TEP_bf, U_TEP3_n_bf

function U_TEP_bf(F2, F3, u)    
    #Calculate potential energy
    U2 = 0.5*((transpose(u) * F2) * u) # ~560 μs with 256 atoms
    U3 = zero(U2)

    for k in eachindex(u)
        for j in eachindex(u)
            for i in eachindex(u)
                U3 += F3[i, j, k] * (u[i]*u[j]*u[k])
            end
        end
    end


    return U2 + U3/6
end

function U_TEP3_bf(F3,u)

    U3 = 0.0

    for j in eachindex(u)
        for i in eachindex(u)
            for k in eachindex(u)
                U3 += F3[i, j, k] * (u[i]*u[j]*u[k])
            end
        end
    end
    return U3/6
end


"""
Energy of element n in third order part of TEP
"""
function U_TEP3_n_bf(F3, u, n)
    F3_view = view(F3, n, :, :)
    u_n = u[n]
    @tensor begin
        U3 = u_n*F3_view[j,k]*u[j]*u[k]
    end
    return U3/6
end