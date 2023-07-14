module ModalAnalysis

using CairoMakie
using DataFrames
using TensorOperations
using LoopVectorization
using JLD2
using StatsBase

#TODO: register online
using ForceConstants

# Code to evaluate TEP
module TEP
    using TensorOperations
    using ForceConstants
    using LoopVectorization

    include("./TEP/TEP.jl")
end
using .TEP

include("DumpParser.jl")
include("MA.jl")
include("INMA.jl")
include("NMA.jl")
include("PostProcess.jl")
# include("WorkSplittingStrategies.jl")



end