using ExplainableAI
using Test
using ADTypes: AutoForwardDiff
using Distributions: Normal
using StableRNGs: StableRNG

@testset "Optional arguments" begin
    distribution = Normal(0.0f0, 1.0f0)
    rng = StableRNG(123)

    @test_nowarn SmoothGrad(identity, 50, distribution)
    @test_nowarn SmoothGrad(identity, 50, distribution, rng)

    gradient_analyzer = Gradient(identity)
    pooling = NormPooling()
    @test_nowarn NoiseAugmentation(gradient_analyzer, 50; pooling)
    @test_nowarn NoiseAugmentation(gradient_analyzer, 50, 0.1f0; pooling)
    @test_nowarn NoiseAugmentation(gradient_analyzer, 50, distribution; pooling)
    @test_nowarn NoiseAugmentation(gradient_analyzer, 50, distribution, rng; pooling)
    @test_nowarn InterpolationAugmentation(gradient_analyzer, 50; pooling)
end

@testset "Augmentations require a pooling" begin
    gradient_analyzer = Gradient(identity)
    @test_throws UndefKeywordError NoiseAugmentation(gradient_analyzer, 50)
    @test_throws UndefKeywordError InterpolationAugmentation(gradient_analyzer, 50)
end

@testset "AD backend selection" begin
    distribution = Normal(0.0f0, 1.0f0)
    rng = StableRNG(123)
    ad = AutoForwardDiff()

    # Analyzers default to the package-wide default backend...
    @test backend(Gradient(identity)) == ExplainableAI.DEFAULT_AD_BACKEND
    @test backend(InputTimesGradient(identity)) == ExplainableAI.DEFAULT_AD_BACKEND
    @test backend(GradCAM(identity, identity)) == ExplainableAI.DEFAULT_AD_BACKEND
    @test backend(SmoothGrad(identity)) == ExplainableAI.DEFAULT_AD_BACKEND
    @test backend(IntegratedGradients(identity)) == ExplainableAI.DEFAULT_AD_BACKEND

    # ...and forward a user-specified backend to the internal `Gradient` analyzer.
    @test backend(Gradient(identity, ad)) == ad
    @test backend(InputTimesGradient(identity, ad)) == ad
    @test backend(GradCAM(identity, identity, ad)) == ad
    @test backend(SmoothGrad(identity; backend = ad)) == ad
    @test backend(SmoothGrad(identity, 50; backend = ad)) == ad
    @test backend(SmoothGrad(identity, 50, distribution; backend = ad)) == ad
    @test backend(SmoothGrad(identity, 50, distribution, rng; backend = ad)) == ad
    @test backend(IntegratedGradients(identity; backend = ad)) == ad
    @test backend(IntegratedGradients(identity, 50; backend = ad)) == ad

    # `backend` also works on manually constructed augmentations
    pooling = NormPooling()
    @test backend(NoiseAugmentation(Gradient(identity, ad), 50; pooling)) == ad
    @test backend(InterpolationAugmentation(Gradient(identity, ad), 50; pooling)) == ad
end
