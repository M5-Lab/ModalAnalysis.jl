export U_TEP3_n_CPU, U_TEP3_n_CUDA, U_TEP3_CUDA,
 TEP_Atomic2, TEP_Modal2, TEP_Atomic3, TEP_Modal3

abstract type Storage end;
struct GPU_Storage <: Storage end
struct CPU_Storage <: Storage end

abstract type TaylorEffectivePotential end;
    
struct TEP_Atomic2{S,E,L} <: TaylorEffectivePotential
    F2::AbstractArray{AbstractFloat,2}
    energy_unit::E
    length_unit::L
end

function TEP_Atomic2(F2)
    F2_storage = (typeof(F2) <: Matrix) ? CPU_Storage : GPU_Storage
    #TODO
end

struct TEP_Atomic3{S,E,L} <: TaylorEffectivePotential
    F2::AbstractArray{AbstractFloat,2}
    F3::AbstractArray{AbstractFloat,3}
    energy_unit::E
    length_unit::L
end

function TEP_Atomic2(F2, F3)
    @assert allequal([size(F3); size(F2)]) "F2 and F3 sizes are inconsistent"

    F2_storage = (typeof(F2) <: Matrix) ? CPU_Storage : GPU_Storage
    F3_storage = (typeof(F3) <: Array{T,3} where T) ? CPU_Storage : GPU_Storage
    @assert F2_storage == F3_storage "Both F2 and F3 must be on CPU or GPU"
     #TODO
end


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

#Sends chunks of N x N x block_size to GPU to save memory
# block_size must be a multiple of N
function U_TEP3_n_CUDA(F3::Array{<:AbstractFloat,3}, cu_u::CuArray{Float32,1}, block_size::Integer)
    N = size(F3,1)
    U_n = CUDA.zeros(Float32, N)
    cuF3 = CUDA.zeros(Float32, N, N, block_size)
    for i in 1:block_size:N
        #Move chunk to GPU
        copyto!(cuF3, F3[:,:,i:i+block_size-1])
        U_n_view = view(U_n, i:i+block_size-1)
        @tensor begin
            U_n_view[n] = cuF3[i,j,n]*cu_u[i]*cu_u[j]
        end
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



########### TESTING

# function U_serial(ifc4_sparse, q)

#     val = 0.0
#     for i in eachindex(ifc4_sparse)
#         val +=  ifc4_sparse[i].v*q[ifc4_sparse[i].i]*q[ifc4_sparse[i].j]*q[ifc4_sparse[i].k]
#     end
#     return val/24
# end


# function U_mapreduce(ifc4_sparse, q)
#     f = (f3_val) -> f3_val.v*q[f3_val.i]*q[f3_val.j]*q[f3_val.k]
#     return mapreduce(f, +, ifc4_sparse,init = 0.0)/24
# end

# #Tests
# struct MyType
#     v::Float64
#     i::Int32
#     j::Int32
#     k::Int32
# end


# N_modes = 768
# N = 30000000
# q = rand(Float64, N_modes)
# random_numbers = rand(Float64,N);
# random_idxs = Int32.(rand(1:N_modes,(N,3)))
# sparse_test = [MyType(random_numbers[i], random_idxs[i,:]...) for i in 1:N]
# sparse_test_SA = StructArray{MyType}(v = random_numbers, i = random_idxs[:,1], j = random_idxs[:,2], k = random_idxs[:,3])



#Optimal method for calculating U_TEP:
# Float64, Sparse StructARray,  CPU (transfer to GPU ~ CPU computation time)
    #takes ~1s to convert 3rd order dense to sparse (ok for NMA, probably fine for INMA??)

# #& add test to check this gives same as plain ole map reduce
# #& check that this is faster than my existing third order
# function U_TEP3_sparse(ifc3_sparse::StructArray{FC_val{Float64,3}}, q::AbstractVector{Float64};
#     nthreads::Integer = Threads.nthreads())
    
#     chunk_size = cld(length(ifc4),nthreads)
#     @assert chunk_size*nthreads >= length(ifc3_sparse) 
#     f = (f3_val) -> f3_val.v*q[f3_val.i]*q[f3_val.j]*q[f3_val.k]

#     partial_sums = zeros(eltype(q), nthreads)
#     Threads.@sync for n in 1:nthreads
#         ifc_view = view(ifc4, (n-1)*chunk_size + 1 : min(n*chunk_size, length(ifc4)))
#         Threads.@spawn begin
#             partial_sums[n] = mapreduce(f, +, ifc_view, init = 0.0)
#         end
#     end

#     return sum(partial_sums)/6

# end

# function U_TEP4_sparse(ifc4_sparse::StructArray{FC_val{Float64,4}}, q::AbstractVector{Float64};
#     nthreads::Integer = Threads.nthreads())
    
#     chunk_size = cld(length(ifc4),nthreads)
#     @assert chunk_size*nthreads >= length(ifc4_sparse) 
#     f = (f4_val) -> f4_val.v*q[f4_val.i]*q[f4_val.j]*q[f4_val.k]*q[f4_val.l]

#     partial_sums = zeros(eltype(q), nthreads)
#     Threads.@sync for n in 1:nthreads
#         ifc_view = view(ifc4, (n-1)*chunk_size + 1 : min(n*chunk_size, length(ifc4)))
#         Threads.@spawn begin
#             partial_sums[n] = mapreduce(f, +, ifc_view, init = 0.0)
#         end
#     end

#     return sum(partial_sums)/24

# end