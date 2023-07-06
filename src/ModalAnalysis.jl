module ModalAnalysis

using DataFrames
using TensorOperations

include("DumpParser.jl")
include("INMA.jl")
include("NMA.jl")

#Remove once this is on registry
include(raw"C:\Users\ejmei\repos\ForceConstants\src\ForceConstants.jl")
using .ForceConstants

# Code to evaluate TEP
module TEP
    include("./TEP/TEP.jl")
end
using .TEP

end