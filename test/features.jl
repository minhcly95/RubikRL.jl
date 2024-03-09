@testset "Features" begin
    @testset "Size test" begin
        # The number of features of a Cube is 54 cells * 6 colors per cell
        @test size(features(rand(Cube))) == (54, 6)

        # Batch test
        for batch in 1:10
            @test size(features(rand(Cube, batch))) == (54, 6, batch)
        end
    end

    @testset "Identity test" begin
        identity_features = vcat(Matrix(I, 6, 6), repeat(Matrix(I, 6, 6), inner=(4, 1), outer=(2, 1)))
        @test features(Cube()) == identity_features
    end
end
