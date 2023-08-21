using LoopVectorization
using ExplainableAI
using ExplainableAI: lrp!, modify_input, modify_denominator, is_compatible
using ExplainableAI: modify_parameters, modify_weight, modify_bias, modify_layer
using ExplainableAI: stabilize_denom
using Flux
using LinearAlgebra: I
using ReferenceTests
using Random

# Fixed pseudo-random numbers
T = Float32
pseudorandn(dims...) = randn(MersenneTwister(123), T, dims...)

const RULES = Dict(
    "ZeroRule"             => ZeroRule(),
    "EpsilonRule"          => EpsilonRule(),
    "GammaRule"            => GammaRule(),
    "ZBoxRule"             => ZBoxRule(0.0f0, 1.0f0),
    "AlphaBetaRule"        => AlphaBetaRule(2.0f0, 1.0f0),
    "WSquareRule"          => WSquareRule(),
    "FlatRule"             => FlatRule(),
    "ZPlusRule"            => ZPlusRule(),
    "GeneralizedGammaRule" => GeneralizedGammaRule(),
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
    modified_layer = nothing

    R̂ₖ = similar(aₖ) # will be inplace updated
    @inferred lrp!(R̂ₖ, rule, layer, modified_layer, aₖ, Rₖ₊₁)
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
    @inferred lrp!(R̂ₖ, rule, layer, modified_layer, aₖ, Rₖ₊₁)
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
    @inferred lrp!(R̂ₖ, rule, layer, modified_layers, aₖ, Rₖ₊₁)
    @test R̂ₖ ≈ Rₖ_α1β0

    rule = AlphaBetaRule(2.0f0, 1.0f0)
    modified_layers = modify_layer(rule, layer)
    @inferred lrp!(R̂ₖ, rule, layer, modified_layers, aₖ, Rₖ₊₁)
    @test R̂ₖ ≈ Rₖ_α2β1

    rule = ZPlusRule()
    modified_layers = modify_layer(rule, layer)
    @inferred lrp!(R̂ₖ, rule, layer, modified_layers, aₖ, Rₖ₊₁)
    @test R̂ₖ ≈ Rₖ_α1β0
end

@testset "GeneralizedGammaRule analytic" begin
    a = [-1.0, 1.0]
    a⁺ = [0.0, 1.0]
    a⁻ = [-1.0, 0.0]
    W = [1.0 -4.0; 2.0 0.0]
    b = [-2.0, 3.0]
    layer = Dense(W, b, leakyrelu) # leakyrelu defaults to a=0.01
    Rₖ₊₁ = [-0.07; 1.0]
    Rₖ₊₁⁺ = [0.0; 1.0]
    Rₖ₊₁⁻ = [-0.07; 0.0]
    @test Rₖ₊₁ == layer(a)

    W⁺ = [1.25 -4.0; 2.5 0.0] # W + γW⁺
    b⁺ = [-2.0, 3.75]         # b + γb⁺
    W⁻ = [1.0 -5.0; 2.0 0.0]  # W + γW⁻
    b⁻ = [-2.5, 3.0]          # b + γb⁻
    sˡ = Rₖ₊₁⁺ ./ stabilize_denom(W⁺ * a⁺ + W⁻ * a⁻ + b⁺, 1.0e-9)
    sʳ = Rₖ₊₁⁻ ./ stabilize_denom(W⁺ * a⁻ + W⁻ * a⁺ + b⁻, 1.0e-9)
    Rₖ =
        a⁺ .* (transpose(W⁺) * sˡ + transpose(W⁻) * sʳ) +
        a⁻ .* (transpose(W⁻) * sˡ + transpose(W⁺) * sʳ)

    rule = GeneralizedGammaRule(0.25)
    ml = modify_layer(rule, layer)
    @test ml.layerˡ⁺.weight == W⁺
    @test ml.layerˡ⁻.weight == W⁻
    @test ml.layerʳ⁻.weight == W⁻
    @test ml.layerʳ⁺.weight == W⁺
    @test ml.layerˡ⁺.bias == b⁺
    @test ml.layerʳ⁻.bias == b⁻
    @test iszero(ml.layerˡ⁻.bias)
    @test iszero(ml.layerʳ⁺.bias)

    R̂ₖ = similar(Rₖ)
    lrp!(R̂ₖ, rule, layer, ml, a, Rₖ₊₁)
    @test R̂ₖ ≈ Rₖ
end

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

    modified_layer = modify_layer(Val(:keep_positive), layer; keep_bias=false)
    @test modified_layer.weight ≈ [1.0 0.0; 2.0 0.0]
    @test modified_layer.bias ≈ [0.0, 0.0]

    modified_layer = modify_layer(Val(:keep_negative), layer)
    @test modified_layer.weight ≈ [0.0 -1.0; 0.0 0.0]
    @test modified_layer.bias ≈ [-1.0, 0.0]

    modified_layer = modify_layer(Val(:keep_negative), layer; keep_bias=false)
    @test modified_layer.weight ≈ [0.0 -1.0; 0.0 0.0]
    @test modified_layer.bias ≈ [0.0, 0.0]

    W = @inferred modify_weight(rule, W)
    b = @inferred modify_bias(rule, b)
    @test W ≈ [1.42 -1.0; 2.84 0.0]
    @test b ≈ [-1.0, 1.42]
end

function run_rule_tests(rule, layer, rulename, layername, aₖ)
    if is_compatible(rule, layer)
        Rₖ₊₁ = layer(aₖ)
        Rₖ = similar(aₖ)
        modified_layer = modify_layer(rule, layer)
        lrp!(Rₖ, rule, layer, modified_layer, aₖ, Rₖ₊₁)
        @test typeof(Rₖ) == typeof(aₖ)
        @test size(Rₖ) == size(aₖ)
        @test_reference "references/rules/$rulename/$layername.jld2" Dict("R" => Rₖ) by =
            (r, a) -> isapprox(r["R"], a["R"]; atol=1e-5, rtol=0.02)
    end
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
                    run_rule_tests(rule, layer, rulename, layername, aₖ_dense)
                end
            end
        end
    end
end

## Test ConvLayers and others
cin, cout = 2, 4
insize = (6, 6, 2, batchsize)
aₖ = pseudorandn(insize)
layers = Dict(
    "Conv"             => Conv((3, 3), cin => cout; init=pseudorandn, bias=pseudorandn(cout)),
    "Conv_relu"        => Conv((3, 3), cin => cout, relu; init=pseudorandn, bias=pseudorandn(cout)),
    "ConvTranspose"    => ConvTranspose((3, 3), cin => cout; init=pseudorandn, bias=pseudorandn(cout)),
    "CrossCor"         => CrossCor((3, 3), cin => cout; init=pseudorandn, bias=pseudorandn(cout)),
    "MaxPool"          => MaxPool((3, 3)),
    "MeanPool"         => MaxPool((3, 3)),
    "AdaptiveMaxPool"  => AdaptiveMaxPool((3, 3)),
    "GlobalMaxPool"    => GlobalMaxPool(),
    "AdaptiveMeanPool" => AdaptiveMeanPool((3, 3)),
    "GlobalMeanPool"   => GlobalMeanPool(),
    "flatten"          => Flux.flatten,
    "Dropout"          => Dropout(0.2; active=false),
    "AlphaDropout"     => AlphaDropout(0.2; active=false),
)
@testset "Other Layers" begin
    for (rulename, rule) in RULES
        @testset "$rulename" begin
            for (layername, layer) in layers
                @testset "$layername" begin
                    run_rule_tests(rule, layer, rulename, layername, aₖ)
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
lrp!(Rₖ_z⁺, rule, layer, modified_layers, aₖ, Rₖ₊₁)
rule = AlphaBetaRule(1.0f0, 0.0f0)
modified_layers = modify_layer(rule, layer)
lrp!(Rₖ_αβ, rule, layer, modified_layers, aₖ, Rₖ₊₁)
@test Rₖ_z⁺ ≈ Rₖ_αβ
