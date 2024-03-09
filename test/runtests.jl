using RubikRL
using Test
using RubikCore, Flux, LinearAlgebra

@testset "RubikRL.jl" begin
    include("features.jl")
    include("rubik_conv.jl")
end
