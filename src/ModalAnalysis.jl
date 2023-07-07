module ModalAnalysis

using DataFrames
using TensorOperations

#TODO: register online
using ForceConstants

# Code to evaluate TEP
module TEP
    include("./TEP/TEP.jl")
end
using .TEP

include("DumpParser.jl")
include("MA.jl")
include("INMA.jl")
include("NMA.jl")
# include("PostProcess.jl")
# include("WorkSplittingStrategies.jl")



end