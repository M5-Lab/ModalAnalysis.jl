export U_TEP3_n, U_TEP3_n_CUDA, U_TEP3_n_CUDA_sparse

function U_TEP3_n(F3::Array{T,3}, u, n) where T
    F3_view = view(F3, :, :, n)
    u_n = u[n]
    @tensor begin
        U3 = F3_view[j,k]*u[j]*u[k]
    end
    return u_n*U3/6
end

function U_TEP3_n_CUDA_sparse(cuF3::CuArray{Float32,3}, cu_u::CuArray{Float32,1}, u::AbstractVector, N_modes)
    cu_u_transpose = transpose(cu_u)
    eng = @views map(n -> cu_u_transpose * (sparse(cuF3[:, :, n]) * cu_u), 1:N_modes)
    return (eng .* u)./6
end

function U_TEP3_n_CUDA(cuF3::CuArray{Float32,3}, cu_u::CuArray{Float32,1}, u::AbstractVector, N_modes)
    cu_u_transpose = transpose(cu_u)
    eng = @views map(n -> cu_u_transpose * (cuF3[:, :, n] * cu_u), 1:N_modes)
    return (eng .* u)./6
end

