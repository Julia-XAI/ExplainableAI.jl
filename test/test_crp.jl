@testset "CRP analytic" begin
    W1 = [1.0 3.0; 4.0 2.0]
    b1 = [0.0, 1.0]
    d1 = Dense(W1, b1, identity)

    W2 = [2.0 4.0; 3.0 1.0]
    b2 = [1.0, 2.0]
    d2 = Dense(W2, b2, identity)

    model = Chain(d1, d2)
    input = reshape([1.0 2.0], 2, 1)

    layer_index = 1
    concepts = TopNConcepts(1)
    analyzer = CRP(LRP(model), layer_index, concepts)

    # Analytic solution:
    # a¹ = input
    # a²[1] = 1*1 + 3*2 + 0 =  1 +  6 + 0 =  7
    # a²[2] = 4*1 + 2*2 + 1 =  4 +  4 + 1 =  9
    # a³[1] = 2*7 + 4*9 + 1 = 14 + 36 + 1 = 51
    # a³[2] = 3*7 + 1*9 + 2 = 21 +  9 + 2 = 32
    # R³ = [1 0], max output neuron selection, masked to 1
    # R²[1] = 14/51 * 1 +  21/32 * 0 = 14/51
    # R²[2] = 36/51 * 1 +   9/32 * 0 = 36/51
    # R² = [0 36/51], CRP Top 1 concept neuron
    # R¹[1] = 1/7 * 0 + 4/9 * 36/51 = 16//51
    # R¹[2] = 6/7 * 0 + 4/9 * 36/51 = 16//51

    expl = analyzer(input)
    @test expl.val ≈ [16 / 51, 16 / 51]
end

@testset "Concept selectors" begin
    @testset "2D input batches" begin
        R = [
            0.360588  0.180214
            0.721713  0.769733
            0.242516  0.918393
            0.736286  0.907605
        ]

        concepts = TopNConcepts(2)
        c1, c2 = concepts(R)
        @test R[c1[1]] == [0.736286;;]
        @test R[c1[2]] == [0.918393;;]
        @test R[c2[1]] == [0.721713;;]
        @test R[c2[2]] == [0.907605;;]

        concepts = IndexedConcepts(3, 2)
        c1, c2 = concepts(R)
        @test R[c1[1]] == [0.242516;;]
        @test R[c1[2]] == [0.918393;;]
        @test R[c2[1]] == [0.721713;;]
        @test R[c2[2]] == [0.769733;;]
    end
    @testset "4D input batches" begin
        R = [ # 2×2×4×2 Array
            0.521664; 0.717311;;
            0.698516; 0.069322;;;
            0.520093; 0.106578;;
            0.943595; 0.286332;;;
            0.185124; 0.624013;;
            0.628282; 0.068466;;;
            0.953506; 0.687441;;
            0.060545; 0.502339;;;;
            0.663752; 0.270196;;
            0.904576; 0.26379;;;
            0.988973; 0.0832523;;
            0.833007; 0.321842;;;
            0.268384; 0.920477;;
            0.674791; 0.748541;;;
            0.426357; 0.090217;;
            0.378728; 0.543824;;;
        ]

        # Check top activated concepts:

        # julia> sum(R; dims=(1, 2))
        # 1×1×4×2 Array{Float64, 4}:
        # [:, :, 1, 1] = 2.006813 # 2
        # [:, :, 2, 1] = 1.856598
        # [:, :, 3, 1] = 1.505885
        # [:, :, 4, 1] = 2.203831 # 1

        # [:, :, 1, 2] = 2.102314
        # [:, :, 2, 2] = 2.227074 # 2
        # [:, :, 3, 2] = 2.612193 # 1
        # [:, :, 4, 2] = 1.439126

        concepts = TopNConcepts(2)
        c1, c2 = concepts(R)
        @test R[c1[1]] == R[:, :, 4:4, 1:1]
        @test R[c2[1]] == R[:, :, 1:1, 1:1]
        @test R[c1[2]] == R[:, :, 3:3, 2:2]
        @test R[c2[2]] == R[:, :, 2:2, 2:2]

        concepts = IndexedConcepts(3, 2)
        c1, c2 = concepts(R)
        @test R[c1[1]] == R[:, :, 3:3, 1:1]
        @test R[c2[1]] == R[:, :, 2:2, 1:1]
        @test R[c1[2]] == R[:, :, 3:3, 2:2]
        @test R[c2[2]] == R[:, :, 2:2, 2:2]
    end
end
