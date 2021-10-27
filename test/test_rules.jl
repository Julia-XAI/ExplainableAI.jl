using ExplainabilityMethods
using ExplainabilityMethods: RULES, modify_params
using Flux
using LinearAlgebra
using ReferenceTests
using Random

# Fixed pseudo-random numbers
T = Float32
pseudorandn(dims...) = randn(MersenneTwister(123), T, dims...)

## Test individual rules
@testset "modify_params" begin
    W, b = [1.0 -1.0; 2.0 0.0], [-1.0, 1.0]
    ρW, ρb = modify_params(GammaRule(; γ=0.42), W, b)
    @test ρW ≈ [1.42 -1.0; 2.84 0.0]
    @test ρb ≈ [-1.0, 1.42]
end

## Test Dense layer
# Define Dense test input
ins = 20 # input dimension
outs = 10 # output dimension
aₖ = pseudorandn(ins)

layers = Dict(
    "Dense_relu" => Dense(ins, outs, relu; init=pseudorandn),
    "Dense_identity" => Dense(Matrix(I, outs, ins), false, identity),
)
@testset "Dense" begin
    for (rulename, ruletype) in RULES
        rule = ruletype()
        @testset "$rulename" begin
            for (layername, layer) in layers
                @testset "$layername" begin
                    Rₖ₊₁ = layer(aₖ)
                    Rₖ = rule(layer, aₖ, Rₖ₊₁)

                    @test typeof(Rₖ) == typeof(aₖ)
                    @test size(Rₖ) == size(aₖ)

                    # println(Rₖ)
                    if rulename == "Dense_identity"
                        # First `outs` dimensions should propagate
                        # activations as relevances, rest should be ≈ 0.
                        @test Rₖ[1:outs] ≈ aₖ[1:outs]
                        @test all(Rₖ[outs:end] .< 1e-8)
                    end

                    @test_reference "references/rules/$rulename/$layername.jld2" Dict(
                        "R" => Rₖ
                    ) by = (r, a) -> isapprox(r["R"], a["R"]; rtol=0.02)
                end
            end
        end
    end
end

## Test PoolingLayers
insize = (6, 6, 2, 1)
aₖ = pseudorandn(insize)

equalpairs = Dict( # these pairs of layers are all equal
    "AdaptiveMaxPool" => (AdaptiveMaxPool((3, 3)), MaxPool((2, 2); pad=0)),
    "GlobalMaxPool" => (GlobalMaxPool(), MaxPool((6, 6); pad=0)),
    "AdaptiveMeanPool" => (AdaptiveMeanPool((3, 3)), MeanPool((2, 2); pad=0)),
    "GlobalMeanPool" => (GlobalMeanPool(), MeanPool((6, 6); pad=0)),
)

@testset "PoolingLayers" begin
    for (rulename, ruletype) in RULES
        rule = ruletype()
        @testset "$rulename" begin
            for (layername, layers) in equalpairs
                @testset "$layername" begin
                    l1, l2 = layers
                    Rₖ₊₁ = l1(aₖ)
                    Rₖ₊₁ == l2(aₖ)
                    Rₖ = rule(l1, aₖ, Rₖ₊₁)
                    Rₖ == rule(l2, aₖ, Rₖ₊₁)

                    @test typeof(Rₖ) == typeof(aₖ)
                    @test size(Rₖ) == size(aₖ)

                    @test_reference "references/rules/$rulename/$layername.jld2" Dict(
                        "R" => Rₖ
                    ) by = (r, a) -> isapprox(r["R"], a["R"]; rtol=0.02)
                end
            end
        end
    end
end

## Test ConvLayers and others
layers = Dict(
    "Conv" => Conv((3, 3), 2 => 4; init=pseudorandn),
    "DepthwiseConv" => DepthwiseConv((3, 3), 2 => 4; init=pseudorandn),
    "ConvTranspose" => ConvTranspose((3, 3), 2 => 4; init=pseudorandn),
    "CrossCor" => CrossCor((3, 3), 2 => 4; init=pseudorandn),
    "flatten" => flatten,
    "Dropout" => Dropout(0.2),
    "AlphaDropout" => AlphaDropout(0.2),
)
@testset "Other Layers" begin
    for (rulename, ruletype) in RULES
        rule = ruletype()
        @testset "$rulename" begin
            for (layername, layer) in layers
                @testset "$layername" begin
                    Rₖ₊₁ = layer(aₖ)
                    Rₖ = rule(layer, aₖ, Rₖ₊₁)

                    @test typeof(Rₖ) == typeof(aₖ)
                    @test size(Rₖ) == size(aₖ)

                    @test_reference "references/rules/$rulename/$layername.jld2" Dict(
                        "R" => Rₖ
                    ) by = (r, a) -> isapprox(r["R"], a["R"]; rtol=0.02)
                end
            end
        end
    end
end
