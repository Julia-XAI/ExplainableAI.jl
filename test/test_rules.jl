using LoopVectorization
using ExplainableAI
using ExplainableAI: modify_params
import ExplainableAI: modify_layer, lrp!
using Flux
using LinearAlgebra: I
using ReferenceTests
using Random

const RULES = Dict(
    "ZeroRule" => ZeroRule(),
    "EpsilonRule" => EpsilonRule(),
    "GammaRule" => GammaRule(),
    "ZBoxRule" => ZBoxRule(0.0f0, 1.0f0),
)

## Hand-written tests
@testset "ZeroRule analytic" begin
    rule = ZeroRule()

    ## Simple dense layer
    Rₖ₊₁ = reshape([1 / 3 2 / 3], 2, 1)
    aₖ = reshape([1.0 2.0;], 2, 1)
    W = [3.0 4.0; 5.0 6.0]
    b = [7.0, 8.0]
    Rₖ = reshape([17 / 90, 316 / 675], 2, 1) # expected output

    layer = Dense(W, b, relu)
    R̂ₖ = similar(aₖ) # will be inplace updated
    @inferred lrp!(R̂ₖ, rule, layer, aₖ, Rₖ₊₁)
    @test R̂ₖ ≈ Rₖ

    ## Pooling layer
    Rₖ₊₁ = Float32.([1 2; 3 4]//30)
    aₖ = Float32.([1 2 3; 10 5 6; 7 8 9])
    Rₖ = Float32.([0 0 0; 4 0 2; 0 0 4]//30) # expected output

    # Repeat in color channel dim and add batch dim
    Rₖ₊₁ = reshape(repeat(Rₖ₊₁, 1, 3), 2, 2, 3, 1)
    aₖ = reshape(repeat(aₖ, 1, 3), 3, 3, 3, 1)
    Rₖ = reshape(repeat(Rₖ, 1, 3), 3, 3, 3, 1)

    layer = MaxPool((2, 2); stride=(1, 1))
    R̂ₖ = similar(aₖ) # will be inplace updated
    @inferred lrp!(R̂ₖ, rule, layer, aₖ, Rₖ₊₁)
    @test R̂ₖ ≈ Rₖ
end

# Fixed pseudo-random numbers
T = Float32
pseudorandn(dims...) = randn(MersenneTwister(123), T, dims...)

## Test individual rules
@testset "modify_params" begin
    W, b = [1.0 -1.0; 2.0 0.0], [-1.0, 1.0]
    ρW, ρb = @inferred modify_params(GammaRule(; γ=0.42), W, b)
    @test ρW ≈ [1.42 -1.0; 2.84 0.0]
    @test ρb ≈ [-1.0, 1.42]
end

## Test Dense layer
# Define Dense test input
ins_dense = 20 # input dimension
outs_dense = 10 # output dimension
batchsize = 2
aₖ_dense = pseudorandn(ins_dense, batchsize)

layers = Dict(
    "Dense_relu" => Dense(ins_dense, outs_dense, relu; init=pseudorandn),
    "Dense_identity" => Dense(Matrix(I, outs_dense, ins_dense), false, identity),
)
@testset "Dense" begin
    for (rulename, rule) in RULES
        @testset "$rulename" begin
            for (layername, layer) in layers
                @testset "$layername" begin
                    Rₖ₊₁ = layer(aₖ_dense)
                    Rₖ = similar(aₖ_dense)
                    @inferred lrp!(Rₖ, rule, layer, aₖ_dense, Rₖ₊₁)

                    @test typeof(Rₖ) == typeof(aₖ_dense)
                    @test size(Rₖ) == size(aₖ_dense)

                    if rulename == "Dense_identity"
                        # First `outs_dense` dimensions should propagate
                        # activations as relevances, rest should be ≈ 0.
                        @test Rₖ[1:outs_dense] ≈ aₖ_dense[1:outs_dense]
                        @test all(Rₖ[outs_dense:end] .< 1e-8)
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
insize = (6, 6, 2, batchsize)
aₖ = pseudorandn(insize)

equalpairs = Dict( # these pairs of layers are all equal
    "AdaptiveMaxPool" => (AdaptiveMaxPool((3, 3)), MaxPool((2, 2); pad=0)),
    "GlobalMaxPool" => (GlobalMaxPool(), MaxPool((6, 6); pad=0)),
    "AdaptiveMeanPool" => (AdaptiveMeanPool((3, 3)), MeanPool((2, 2); pad=0)),
    "GlobalMeanPool" => (GlobalMeanPool(), MeanPool((6, 6); pad=0)),
)

@testset "PoolingLayers" begin
    for (rulename, rule) in RULES
        @testset "$rulename" begin
            for (layername, layers) in equalpairs
                @testset "$layername" begin
                    l1, l2 = layers
                    Rₖ₊₁ = l1(aₖ)
                    @test Rₖ₊₁ == l2(aₖ)
                    Rₖ1 = similar(aₖ)
                    Rₖ2 = similar(aₖ)
                    @inferred lrp!(Rₖ1, rule, l1, aₖ, Rₖ₊₁)
                    @inferred lrp!(Rₖ2, rule, l2, aₖ, Rₖ₊₁)
                    @test Rₖ1 == Rₖ2

                    @test typeof(Rₖ1) == typeof(aₖ)
                    @test size(Rₖ1) == size(aₖ)

                    @test_reference "references/rules/$rulename/$layername.jld2" Dict(
                        "R" => Rₖ1
                    ) by = (r, a) -> isapprox(r["R"], a["R"]; rtol=0.02)
                end
            end
        end
    end
end

## Test ConvLayers and others
layers = Dict(
    "Conv" => Conv((3, 3), 2 => 4; init=pseudorandn),
    "MaxPool" => MaxPool((3, 3)),
    "MeanPool" => MaxPool((3, 3)),
    "ConvTranspose" => ConvTranspose((3, 3), 2 => 4; init=pseudorandn),
    "CrossCor" => CrossCor((3, 3), 2 => 4; init=pseudorandn),
    "flatten" => flatten,
    "Dropout" => Dropout(0.2),
    "AlphaDropout" => AlphaDropout(0.2),
)
@testset "Other Layers" begin
    for (rulename, rule) in RULES
        @testset "$rulename" begin
            for (layername, layer) in layers
                @testset "$layername" begin
                    Rₖ₊₁ = layer(aₖ)
                    Rₖ = similar(aₖ)
                    @inferred lrp!(Rₖ, rule, layer, aₖ, Rₖ₊₁)

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

## Test custom layers & default AD fallback using the ZeroRule
# Compare with references of non-wrapped layers
struct TestWrapper{T}
    layer::T
end
(w::TestWrapper)(x) = w.layer(x)
modify_layer(r::AbstractLRPRule, w::TestWrapper) = modify_layer(r, w.layer)
lrp!(Rₖ, rule::ZBoxRule, w::TestWrapper, aₖ, Rₖ₊₁) = lrp!(Rₖ, rule, w.layer, aₖ, Rₖ₊₁)

layers = Dict(
    "Conv" => (Conv((3, 3), 2 => 4; init=pseudorandn), aₖ),
    "Dense_relu" => (Dense(ins_dense, outs_dense, relu; init=pseudorandn), aₖ_dense),
    "flatten" => (flatten, aₖ),
)
@testset "Custom layers" begin
    for (rulename, rule) in RULES
        @testset "$rulename" begin
            for (layername, (layer, aₖ)) in layers
                @testset "$layername" begin
                    wrapped_layer = TestWrapper(layer)
                    Rₖ₊₁ = wrapped_layer(aₖ)
                    Rₖ = similar(aₖ)
                    @inferred lrp!(Rₖ, rule, wrapped_layer, aₖ, Rₖ₊₁)

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
