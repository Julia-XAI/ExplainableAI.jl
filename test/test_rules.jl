using LoopVectorization
using ExplainableAI
using ExplainableAI: lrp!, has_weight_and_bias
using ExplainableAI: modify_input, modify_denominator, is_compatible
using ExplainableAI: modify_parameters, modify_weight, modify_bias, modify_layer
using Flux
using LinearAlgebra: I
using ReferenceTests
using Random

const RULES = Dict(
    "ZeroRule"      => ZeroRule(),
    "EpsilonRule"   => EpsilonRule(),
    "GammaRule"     => GammaRule(),
    "ZBoxRule"      => ZBoxRule(0.0f0, 1.0f0),
    "AlphaBetaRule" => AlphaBetaRule(2.0f0, 1.0f0),
    "WSquareRule"   => WSquareRule(),
    "FlatRule"      => FlatRule(),
    "ZPlusRule"     => ZPlusRule(),
)

isa_constant_param_rule(rule) = isa(rule, Union{ZeroRule,EpsilonRule})

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

@testset "AlphaBetaRule analytic" begin
    aₖ = [1.0f0, 1.0f0]
    W = [1.0f0 -1.0f0]
    b = [-1.0f0]
    layer = Dense(W, b, identity)
    Rₖ₊₁ = layer(aₖ)

    # Expected outputs
    Rₖ_α1β0 = [-1.0f0, 0.0f0]
    Rₖ_α2β1 = [-2.0f0, 0.5f0]

    R̂ₖ = similar(aₖ) # will be inplace updated
    rule = AlphaBetaRule(1.0f0, 0.0f0)
    modified_layers = modify_layer(rule, layer)
    @inferred lrp!(R̂ₖ, rule, modified_layers, aₖ, Rₖ₊₁)
    @test R̂ₖ ≈ Rₖ_α1β0

    rule = AlphaBetaRule(2.0f0, 1.0f0)
    modified_layers = modify_layer(rule, layer)
    @inferred lrp!(R̂ₖ, rule, modified_layers, aₖ, Rₖ₊₁)
    @test R̂ₖ ≈ Rₖ_α2β1

    rule = ZPlusRule()
    modified_layers = modify_layer(rule, layer)
    @inferred lrp!(R̂ₖ, rule, modified_layers, aₖ, Rₖ₊₁)
    @test R̂ₖ ≈ Rₖ_α1β0
end

# Fixed pseudo-random numbers
T = Float32
pseudorandn(dims...) = randn(MersenneTwister(123), T, dims...)

## Test individual rules
@testset "modify_parameters" begin
    rule = GammaRule(0.42)
    W, b = [1.0 -1.0; 2.0 0.0], [-1.0, 1.0]
    layer = Dense(W, b, relu)

    modified_layer = modify_layer(rule, layer)
    @test modified_layer.weight ≈ [1.42 -1.0; 2.84 0.0]
    @test modified_layer.bias ≈ [-1.0, 1.42]
    @test layer.weight ≈ W
    @test layer.bias ≈ b

    modified_layer = modify_layer(Val(:keep_positive), layer)
    @test modified_layer.weight ≈ [1.0 0.0; 2.0 0.0]
    @test modified_layer.bias ≈ [0.0, 1.0]

    modified_layer = modify_layer(Val(:keep_positive_zero_bias), layer)
    @test modified_layer.weight ≈ [1.0 0.0; 2.0 0.0]
    @test modified_layer.bias ≈ [0.0, 0.0]

    modified_layer = modify_layer(Val(:keep_negative), layer)
    @test modified_layer.weight ≈ [0.0 -1.0; 0.0 0.0]
    @test modified_layer.bias ≈ [-1.0, 0.0]

    modified_layer = modify_layer(Val(:keep_negative_zero_bias), layer)
    @test modified_layer.weight ≈ [0.0 -1.0; 0.0 0.0]
    @test modified_layer.bias ≈ [0.0, 0.0]

    W = @inferred modify_weight(rule, W)
    b = @inferred modify_bias(rule, b)
    @test W ≈ [1.42 -1.0; 2.84 0.0]
    @test b ≈ [-1.0, 1.42]
end

## Test Dense layer
# Define Dense test input
din = 20 # input dimension
dout = 10 # output dimension
batchsize = 2
aₖ_dense = pseudorandn(din, batchsize)

layers = Dict(
    "Dense_relu"     => Dense(pseudorandn(dout, din), pseudorandn(dout), relu),
    "Dense_identity" => Dense(Matrix{Float32}(I, dout, din), false, identity),
)
@testset "Dense" begin
    for (rulename, rule) in RULES
        @testset "$rulename" begin
            for (layername, layer) in layers
                @testset "$layername" begin
                    Rₖ₊₁ = layer(aₖ_dense)
                    Rₖ = similar(aₖ_dense)
                    modified_layer = modify_layer(rule, layer)
                    @inferred lrp!(Rₖ, rule, modified_layer, aₖ_dense, Rₖ₊₁)

                    @test typeof(Rₖ) == typeof(aₖ_dense)
                    @test size(Rₖ) == size(aₖ_dense)

                    if rulename == "Dense_identity"
                        # First `dout` dimensions should propagate
                        # activations as relevances, rest should be ≈ 0.
                        @test Rₖ[1:dout] ≈ aₖ_dense[1:dout]
                        @test all(Rₖ[dout:end] .< 1e-8)
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
    "AdaptiveMaxPool"  => (AdaptiveMaxPool((3, 3)), MaxPool((2, 2); pad=0)),
    "GlobalMaxPool"    => (GlobalMaxPool(), MaxPool((6, 6); pad=0)),
    "AdaptiveMeanPool" => (AdaptiveMeanPool((3, 3)), MeanPool((2, 2); pad=0)),
    "GlobalMeanPool"   => (GlobalMeanPool(), MeanPool((6, 6); pad=0)),
)

@testset "PoolingLayers" begin
    for (rulename, rule) in RULES
        @testset "$rulename" begin
            for (layername, (l1, l2)) in equalpairs
                if is_compatible(rule, l1) && is_compatible(rule, l2)
                    @testset "$layername" begin
                        ml1 = modify_layer(rule, l1)
                        ml2 = modify_layer(rule, l2)

                        Rₖ₊₁ = l1(aₖ)
                        @test Rₖ₊₁ == l2(aₖ)
                        Rₖ1 = similar(aₖ)
                        Rₖ2 = similar(aₖ)

                        if isa_constant_param_rule(rule)
                            @inferred lrp!(Rₖ1, rule, ml1, aₖ, Rₖ₊₁)
                            @inferred lrp!(Rₖ2, rule, ml2, aₖ, Rₖ₊₁)
                            @test Rₖ1 == Rₖ2
                            @test typeof(Rₖ1) == typeof(aₖ)
                            @test size(Rₖ1) == size(aₖ)
                            @test_reference "references/rules/$rulename/$layername.jld2" Dict(
                                "R" => Rₖ1
                            ) by = (r, a) -> isapprox(r["R"], a["R"]; rtol=0.02)
                        else
                            @test_throws ArgumentError lrp!(Rₖ1, rule, l1, aₖ, Rₖ₊₁)
                        end
                    end
                end
            end
        end
    end
end

## Test ConvLayers and others
cin = 2
cout = 4
layers = Dict(
    "Conv"          => Conv((3, 3), cin => cout; init=pseudorandn, bias=pseudorandn(cout)),
    "Conv_relu"     => Conv((3, 3), cin => cout, relu; init=pseudorandn, bias=pseudorandn(cout)),
    "ConvTranspose" => ConvTranspose((3, 3), cin => cout; init=pseudorandn, bias=pseudorandn(cout)),
    "CrossCor"      => CrossCor((3, 3), cin => cout; init=pseudorandn, bias=pseudorandn(cout)),
    "MaxPool"       => MaxPool((3, 3)),
    "MeanPool"      => MaxPool((3, 3)),
    "flatten"       => Flux.flatten,
    "Dropout"       => Dropout(0.2),
    "AlphaDropout"  => AlphaDropout(0.2),
)
@testset "Other Layers" begin
    for (rulename, rule) in RULES
        @testset "$rulename" begin
            for (layername, layer) in layers
                if is_compatible(rule, layer)
                    @testset "$layername" begin
                        Rₖ₊₁ = layer(aₖ)
                        Rₖ = similar(aₖ)
                        modified_layer = modify_layer(rule, layer)

                        if has_weight_and_bias(layer) || isa_constant_param_rule(rule)
                            @inferred lrp!(Rₖ, rule, modified_layer, aₖ, Rₖ₊₁)
                            @test typeof(Rₖ) == typeof(aₖ)
                            @test size(Rₖ) == size(aₖ)
                            @test_reference "references/rules/$rulename/$layername.jld2" Dict(
                                "R" => Rₖ
                            ) by = (r, a) -> isapprox(r["R"], a["R"]; atol=1e-5, rtol=0.02)
                        else
                            @test_throws ArgumentError lrp!(Rₖ, rule, layer, aₖ, Rₖ₊₁)
                        end
                    end
                end
            end
        end
    end
end

# Test equivalence of ZPlusRule() and AlphaBetaRule(1.0f0, 0.0f0)
layer = layers["Conv"]
Rₖ₊₁ = layer(aₖ)
Rₖ_z⁺ = similar(aₖ)
Rₖ_αβ = similar(aₖ)
rule = ZPlusRule()
modified_layers = modify_layer(rule, layer)
lrp!(Rₖ_z⁺, rule, modified_layers, aₖ, Rₖ₊₁)
rule = AlphaBetaRule(1.0f0, 0.0f0)
modified_layers = modify_layer(rule, layer)
lrp!(Rₖ_αβ, rule, modified_layers, aₖ, Rₖ₊₁)
@test Rₖ_z⁺ ≈ Rₖ_αβ
