module ModalAnalysis

using LinearAlgebra
using CairoMakie
using DataFrames
using TensorOperations
using JLD2, CodecZlib
using StatsBase
using CUDA
using DelimitedFiles
using Unitful

@static if Sys.islinux()
	using ThreadPinning
	pinthreads(:cores)
end


#If add KS test back need HypothesisTests, Images

#TODO: register online
using ForceConstants

module TEP
    using ForceConstants
    using TensorOperations
    using CUDA, cuTENSOR
    using LoopVectorization

    include("./TEP/TEP.jl")
    include("./TEP/DeltaTEP.jl")
end
using .TEP

module MonteCarloHeatCapacity
    using ForceConstants
    using Distributions
    using Unitful
    using StatsBase
    using DataFrames
    using JLD2
    using ..TEP
    include("./MonteCarloHeatCapacity/MonteCarlo.jl")
end
using .MonteCarloHeatCapacity


include("DumpParser.jl")
include("MA.jl")
include("NMA.jl")
include("INMA.jl")
include("PostProcess.jl")
include("AverageINMs.jl")

include("./workflows/GPU_NMA_Job.jl")
include("./workflows/MonteCarloJobs.jl")
include("./workflows/AvgINM_job.jl")

end
