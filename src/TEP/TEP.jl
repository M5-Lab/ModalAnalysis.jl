export U_TEP3_n_CPU, U_TEP3_n_CUDA

function U_TEP3_n_CPU(F3::Array{T,3}, u) where T
    @tensor begin
        U3[n] := F3[j,k,n]*u[j]*u[k]
    end
    return (u .* U3)./6
end


function U_TEP3_n_CUDA(cuF3::CuArray{Float32,3}, cu_u::CuArray{Float32,1})
    @tensor begin
        per_mode[n] := cuF3[i,j,n] * cu_u[i] * cu_u[j]
    end
    return Array((per_mode .* cu_u)./6)
end