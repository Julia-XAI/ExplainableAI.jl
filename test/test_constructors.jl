using ExplainableAI
using Test
using Distributions: Normal
using StableRNGs: StableRNG

@testset "Optional arguments" begin
    distribution = Normal(0.0f0, 1.0f0)
    rng = StableRNG(123)

    @test_nowarn SmoothGrad(identity, 50, distribution)
    @test_nowarn SmoothGrad(identity, 50, distribution, rng)

    gradient_analyzer = Gradient(identity)
    @test_nowarn NoiseAugmentation(gradient_analyzer, 50, distribution)
    @test_nowarn NoiseAugmentation(gradient_analyzer, 50, distribution, rng)
end
