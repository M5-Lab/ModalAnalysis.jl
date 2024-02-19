module ModalAnalysis

using LinearAlgebra
using CairoMakie
using DataFrames
using TensorOperations
using JLD2, CodecZlib
using StatsBase
using CUDA, cuTENSOR
using DelimitedFiles
using TimerOutputs
using Unitful
using TensorOperations
using ThreadPinning
pinthreads(:cores)


#If add KS test back need HypothesisTests, Images

#TODO: register online
using ForceConstants

module TEP
    include("./TEP/TEP.jl")
    include("./TEP/DeltaTEP.jl")
end
using .TEP

#* CAN THIS BE PACKAGE EXTENSION?? SEPARATE PACKAGE ENTIRELY?
module MonteCarloHeatCapacity
    include("./MonteCarloHeatCapacity/MonteCarlo.jl")
end
using .MonteCarloHeatCapacity


include("DumpParser.jl")
include("MA.jl")
include("NMA.jl")
include("INMA.jl")
include("PostProcess.jl")
include("AverageINMs.jl")

include("./workflows/GPU_Job.jl")

end