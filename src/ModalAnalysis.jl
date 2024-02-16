module ModalAnalysis

using LinearAlgebra
using CairoMakie
using DataFrames
using TensorOperations
using JLD2, CodecZlib
using StatsBase
using HypothesisTests
using CUDA, cuTENSOR
using DelimitedFiles
import Images: colorview, Gray, save
using TimerOutputs
using Unitful
using CUDA, cuTENSOR

#TODO: register online
using ForceConstants

#TODO add thread pinning if on Linux

#Comment out KS test stuff (remove hypo tests, Images)

# Code to evaluate TEP
include("./TEP/TEP.jl")

include("DumpParser.jl")
include("MA.jl")
include("NMA.jl")
include("INMA.jl")
include("PostProcess.jl")
include("AverageINMs.jl")

include("./workflows/GPU_Job.jl")

end