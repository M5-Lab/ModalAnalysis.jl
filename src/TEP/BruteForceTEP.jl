"""
Energy of second order part of TEP
"""
function U_TEP2(F2, u)    
    @tensor begin
        U2 = F2[i,j]*u[i]*u[j]
    end
    return U2/2
end

"""
Energy of third order part of TEP
"""
function U_TEP3(F3, u)
    @tensor begin
        U3 = F3[i,j,k]*u[i]*u[j]*u[k]
    end
    return U3/6
end

"""
Energy of element n in third order part of TEP
"""
function U_TEP3_n(F3, u, n)
    F3_view = view(F3, n, :, :)
    u_n = u[n]
    @tensor begin
        U3 = u_n*F3_view[j,k]*u[j]*u[k]
    end
    return U3/6
end

