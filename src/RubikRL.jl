module RubikRL

using RubikCore
using Flux
using Distributions, DataStructures

include("features.jl")
export features

include("rubik_conv.jl")
export RubikConv

end
