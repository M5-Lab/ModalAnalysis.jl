export U_TEP3_n_CPU, U_TEP3_n_CUDA, U_TEP3_CUDA

function U_TEP3_n_CPU(F3::Array{T,3}, u) where T
    @tensor begin
        U_n[n] := F3[j,k,n]*u[j]*u[k]
    end
    U_n .*= (u./6)
    return U_n
end


function U_TEP3_n_CUDA(cuF3::CuArray{Float32,3}, cu_u::CuArray{Float32,1})
    @tensor begin
        U_n[n] := cuF3[i,j,n] * cu_u[i] * cu_u[j]
    end
    U_n .*= (cu_u./6)
    return U_n
end

function U_TEP3_CUDA(cuF3::CuArray{Float32,3}, cu_u::CuArray{Float32,1})
    @tensor begin
        U = cuF3[i,j,k] * cu_u[i] * cu_u[j] * cu_u[k]
    end
    return U/6
end