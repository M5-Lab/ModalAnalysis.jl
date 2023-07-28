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
        gpu = U_TEP3_n_CUDA(CuArray(K3), CuArray(u), u, 3)

        @test isapprox(cpu, gpu, atol = 1e-7)

        brute_force = U_TEP3_bf(K3, u)
        total_gpu = U_TEP3_CUDA(K3, u)

        @test isapprox(brute_force, total_gpu)
        @test isapprox(sum(cpu), brute_force)

        bf = U_TEP3_n_bf(K3, u, 2)
        
        @test isapprox(bf, cpu)
    end
end

@testset "Modal vs Disp Coords" begin

    phi, dynmat, F3, K3, freqs_sq = load("./test_data/TEP.jld2", "phi", "dynmat", "F3", "K3", "freqs_sq")

    base_path = raw"./test_data/"
    equilibrium_data_path = joinpath(base_path, "equilibrium.atom")
    dump_path = joinpath(base_path, "dump.atom")
    thermo_path = joinpath(base_path,"thermo_data.txt")

    eq = LammpsDump(equilibrium_data_path);
    parse_timestep!(eq, 1)
    atom_masses = get_col(eq, "mass")
    mass_sqrt = sqrt.(atom_masses)


    ld = LammpsDump(dump_path);
    dump_file = open(ld.path, "r")
    ld, dump_file = parse_next_timestep!(ld, dump_file)
    ld, dump_file = parse_next_timestep!(ld, dump_file)
    ld, dump_file = parse_next_timestep!(ld, dump_file)
    ld, dump_file = parse_next_timestep!(ld, dump_file)

    disp = Matrix(ld.data_storage[!, ["xu","yu","zu"]]) .- Matrix(eq.data_storage[!, ["x","y","z"]])
    
    F2 = dynmat .* m;
    disp_copy = reduce(vcat, eachrow(copy(disp)))
    U2_disp = 0.5*((transpose(disp_copy) * F2) * disp_copy)
    U3_disp = U_TEP3_bf(F3, disp_copy)

    for col in eachcol(disp) col .*= mass_sqrt end
    disp_mw = reduce(vcat, eachrow(disp))

    q = phi' * disp_mw

    U2_mode = 0.5* sum(freqs_sq .* (q.^2))
    U3_mode = U_TEP3_bf(K3, q)

    @test U_disp ≈ U_mode

    close(dump_file)

end