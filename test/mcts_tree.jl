@testset "MCTS Tree" begin
    @testset "Repeated step" begin
        t = MCTSTree(rand(Cube), Model(1, 1))
        @test playouts(t) == 1
        for i in 1:100
            step!(t)
            # Every step increases playouts by exactly 1
            @test playouts(t) == i + 1
        end
    end
end
