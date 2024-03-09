using RubikRL
using Test
using RubikCore, Flux, LinearAlgebra

# Dummy model for testing purpose
struct DummyModel end
(::DummyModel)(::Cube) = rand() * 10, fill(1/18, 18)

@testset "RubikRL.jl" begin
    include("features.jl")
    include("rubik_conv.jl")
    include("mcts_tree.jl")
end
