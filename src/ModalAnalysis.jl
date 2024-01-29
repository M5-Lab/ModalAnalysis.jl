module ModalAnalysis

using LinearAlgebra
using CairoMakie
using DataFrames
using TensorOperations
using LoopVectorization
using JLD2, CodecZlib
using StatsBase
using HypothesisTests
using CUDA, cuTENSOR
using DelimitedFiles
import Images: colorview, Gray, save
using TimerOutputs

#TODO: register online
using ForceConstants

#TODO add thread pinning if on Linux

# Code to evaluate TEP
module TEP
    using TensorOperations
    using ForceConstants
    using LoopVectorization
    using CUDA, cuTENSOR

    include("./TEP/TEP.jl")
    include("./TEP/BruteForceTEP.jl")
end
using .TEP

include("DumpParser.jl")
include("MA.jl")
include("INMA.jl")
include("NMA.jl")
include("PostProcess.jl")
include("AverageINMs.jl")

end