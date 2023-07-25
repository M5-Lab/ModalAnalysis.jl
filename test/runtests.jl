using Test
using CUDA
using ModalAnalysis

# Allow CUDA device to be specified
const DEVICE = get(ENV, "DEVICE", "0")
run_gpu_tests = false
if CUDA.functional()
    device!(parse(Int, DEVICE))
    run_gpu_tests = true
    @info "The GPU tests will be run on device $DEVICE"
else
    @warn "The GPU tests will not be run as a CUDA-enabled device is not available"
end


@testset "TEP Per Mode" begin
    if run_gpu_tests
        K3 = zeros(Float32, (3,3,3))
        K3[:,:,1] = [0 0 3; 0 1 0; 3 0 0]
        K3[:,:,2] = [0 1 0; 1 13 0; 0 0 0]
        K3[:,:,3] = [3 0 0; 0 0 0; 0 0 0]

        u = rand(Float32, 3)
        
        cpu = U_TEP3_n_CPU(K3, u)
        gpu = U_TEP3_n_CUDA(CuArray(K3), CuArray(u), u::AbstractVector, 3)

        @test isapprox(cpu, gpu, atol = 1e-7)

        brute_force = U_TEP3_bf(K3, u)

        @test isapprox(sum(cpu), brute_force)
    end
end