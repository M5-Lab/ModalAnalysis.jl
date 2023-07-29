module ModalAnalysis

using LinearAlgebra
using CairoMakie
using DataFrames
using TensorOperations
using LoopVectorization
using JLD2
using StatsBase
using ProgressMeter
using CUDA, cuTENSOR

#TODO: register online
using ForceConstants

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
# include("WorkSplittingStrategies.jl")

end