@testset "RubikConv" begin
    @testset "Size test" begin
        for _ in 1:10
            cin, cout, batch = rand(1:64, 3)
            a = RubikConv(cin => cout)
            x = rand(Float32, 54, cin, batch)
            y = a(x)
            @test size(y) == (54, cout, batch)
        end
    end

    @testset "Connection test" begin
        cin, cout, batch = 32, 32, 32
        a = RubikConv(cin => cout)
        fill!(a.weight, 1)

        @testset "Center" begin
            for (i, row) in enumerate(eachrow(RubikRL.CENTER_WINDOW))
                not_row = setdiff(1:54, row)

                x = zeros(Float32, 54, cin, batch)
                x[i, 1, :] .= 1
                y = a(x)

                @test all(y[row, :, :] .!== 0)
                @test all(y[not_row, :, :] .== 0)
            end
        end

        @testset "Edge" begin
            for (i, row) in enumerate(eachrow(RubikRL.EDGE_WINDOW))
                not_row = setdiff(1:54, row)

                x = zeros(Float32, 54, cin, batch)
                x[6 + i, 1, :] .= 1
                y = a(x)

                @test all(y[row, :, :] .!== 0)
                @test all(y[not_row, :, :] .== 0)
            end
        end

        @testset "Corner" begin
            for (i, row) in enumerate(eachrow(RubikRL.CORNER_WINDOW))
                not_row = setdiff(1:54, row)

                x = zeros(Float32, 54, cin, batch)
                x[30 + i, 1, :] .= 1
                y = a(x)

                @test all(y[row, :, :] .!== 0)
                @test all(y[not_row, :, :] .== 0)
            end
        end
    end
end
