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
    Rᵏ⁺¹ = reshape([1 / 3 2 / 3], 2, 1)
    aᵏ = reshape([1.0 2.0], 2, 1)
    W = [3.0 4.0; 5.0 6.0]
    b = [7.0, 8.0]
    Rᵏ = reshape([17 / 90, 316 / 675], 2, 1) # expected output

    layer = Dense(W, b, relu)
    modified_layer = nothing

    R̂ᵏ = similar(aᵏ) # will be inplace updated
    @inferred lrp!(R̂ᵏ, rule, layer, modified_layer, aᵏ, Rᵏ⁺¹)
    @test R̂ᵏ ≈ Rᵏ

    ## Pooling layer
    Rᵏ⁺¹ = Float32.([1 2; 3 4]//30)
    aᵏ = Float32.([1 2 3; 10 5 6; 7 8 9])
    Rᵏ = Float32.([0 0 0; 4 0 2; 0 0 4]//30) # expected output

    # Repeat in color channel dim and add batch dim
    Rᵏ⁺¹ = reshape(repeat(Rᵏ⁺¹, 1, 3), 2, 2, 3, 1)
    aᵏ = reshape(repeat(aᵏ, 1, 3), 3, 3, 3, 1)
    Rᵏ = reshape(repeat(Rᵏ, 1, 3), 3, 3, 3, 1)

    layer = MaxPool((2, 2); stride=(1, 1))
    R̂ᵏ = similar(aᵏ) # will be inplace updated
    @inferred lrp!(R̂ᵏ, rule, layer, modified_layer, aᵏ, Rᵏ⁺¹)
    @test R̂ᵏ ≈ Rᵏ
end

@testset "Parallel analytic" begin
    W = [3.0 4.0; 5.0 6.0]
    b = [7.0, 8.0]
    model = Chain(Parallel(+, identity, Dense(W, b, relu)))
    composite = Composite(
        GlobalTypeMap(typeof(identity) => PassRule(), Dense => ZeroRule())
    )

    aᵏ = reshape([1.0 2.0], 2, 1)
    # aᵏ⁺¹₁ = identity(aᵏ) = [1 2]
    # aᵏ⁺¹₂ = Dense(aᵏ) = [3*1 + 4*2 + 7,  5*1 + 6*2 + 8] = [18 25]
    # aᵏ⁺¹ = [19 27]

    # For output neuron 1:
    # Rᵏ⁺¹ = [1 0]
    # Rᵏ⁺¹₁ = [1 0] .* [ 1  2] ./ [19 27] = [ 1/19 0]
    # Rᵏ⁺¹₂ = [1 0] .* [18 25] ./ [19 27] = [18/19 0]
    # The identity function is trivial:
    # Rᵏ₁ = Rᵏ⁺¹₁ = [1/19 0]
    # The Dense layer requires computation of LRP:
    # [Rᵏ₂]ⱼ = ∑ᵢ ([W]ᵢⱼ * [aᵏ]ⱼ / [aᵏ⁺¹₂]ᵢ *  [Rᵏ⁺¹₂]ᵢ)
    # [Rᵏ₂]₁ = 3*1/18*(18/19) + 5*1/25*0 = 3/19
    # [Rᵏ₂]₂ = 4*2/18*(18/19) + 6*2/25*0 = 8/19
    # Rᵏ₂ = [3/19 8/19]
    # Rᵏ = Rᵏ₁ + Rᵏ₂ = [4/19 8/19]
    e1 = analyze(aᵏ, LRP(model, composite), 1)
    @test e1.val ≈ reshape([4 / 19 8 / 19], 2, 1)

    # Analogous for output neuron 2:
    # Rᵏ⁺¹ = [0 1]
    # Rᵏ⁺¹₁ = [0 1] .* [ 1  2] ./ [19 27] = [0  2/27]
    # Rᵏ⁺¹₂ = [0 1] .* [18 25] ./ [19 27] = [0 25/27]
    # Identity function:
    # Rᵏ₁ = Rᵏ⁺¹₁ = [0 2/27]
    # Dense layer:
    # [Rᵏ₂]ⱼ = ∑ᵢ ([W]ᵢⱼ * [aᵏ]ⱼ / [aᵏ⁺¹₂]ᵢ *  [Rᵏ⁺¹₂]ᵢ)
    # [Rᵏ₂]₁ = 3*1/18*0 + 5*1/25*(25/27) =  5/27
    # [Rᵏ₂]₂ = 4*2/18*0 + 6*2/25*(25/27) = 12/27
    # Rᵏ₂ = [5/27 12/27]
    # Rᵏ = Rᵏ₁ + Rᵏ₂ = [5/27 14/27]
    e2 = analyze(aᵏ, LRP(model, composite), 2)
    @test e2.val ≈ reshape([5 / 27 14 / 27], 2, 1)
end

@testset "AlphaBetaRule analytic" begin
    aᵏ = [1.0f0, 1.0f0]
    W = [1.0f0 -1.0f0]
    b = [-1.0f0]
    layer = Dense(W, b, identity)
    Rᵏ⁺¹ = layer(aᵏ)

    # Expected outputs
    Rᵏ_α1β0 = [-1.0f0, 0.0f0]
    Rᵏ_α2β1 = [-2.0f0, 0.5f0]

    R̂ᵏ = similar(aᵏ) # will be inplace updated
    rule = AlphaBetaRule(1.0f0, 0.0f0)
    modified_layers = modify_layer(rule, layer)
    @inferred lrp!(R̂ᵏ, rule, layer, modified_layers, aᵏ, Rᵏ⁺¹)
    @test R̂ᵏ ≈ Rᵏ_α1β0

    rule = AlphaBetaRule(2.0f0, 1.0f0)
    modified_layers = modify_layer(rule, layer)
    @inferred lrp!(R̂ᵏ, rule, layer, modified_layers, aᵏ, Rᵏ⁺¹)
    @test R̂ᵏ ≈ Rᵏ_α2β1

    rule = ZPlusRule()
    modified_layers = modify_layer(rule, layer)
    @inferred lrp!(R̂ᵏ, rule, layer, modified_layers, aᵏ, Rᵏ⁺¹)
    @test R̂ᵏ ≈ Rᵏ_α1β0
end

@testset "GeneralizedGammaRule analytic" begin
    a = [-1.0, 1.0]
    a⁺ = [0.0, 1.0]
    a⁻ = [-1.0, 0.0]
    W = [1.0 -4.0; 2.0 0.0]
    b = [-2.0, 3.0]
    layer = Dense(W, b, leakyrelu) # leakyrelu defaults to a=0.01
    Rᵏ⁺¹ = [-0.07; 1.0]
    Rᵏ⁺¹⁺ = [0.0; 1.0]
    Rᵏ⁺¹⁻ = [-0.07; 0.0]
    @test Rᵏ⁺¹ == layer(a)

    W⁺ = [1.25 -4.0; 2.5 0.0] # W + γW⁺
    b⁺ = [-2.0, 3.75]         # b + γb⁺
    W⁻ = [1.0 -5.0; 2.0 0.0]  # W + γW⁻
    b⁻ = [-2.5, 3.0]          # b + γb⁻
    sˡ = Rᵏ⁺¹⁺ ./ stabilize_denom(W⁺ * a⁺ + W⁻ * a⁻ + b⁺, 1.0e-9)
    sʳ = Rᵏ⁺¹⁻ ./ stabilize_denom(W⁺ * a⁻ + W⁻ * a⁺ + b⁻, 1.0e-9)
    Rᵏ =
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

    R̂ᵏ = similar(Rᵏ)
    lrp!(R̂ᵏ, rule, layer, ml, a, Rᵏ⁺¹)
    @test R̂ᵏ ≈ Rᵏ
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

function run_rule_tests(rule, layer, rulename, layername, aᵏ)
    if is_compatible(rule, layer)
        Rᵏ⁺¹ = layer(aᵏ)
        Rᵏ = similar(aᵏ)
        modified_layer = modify_layer(rule, layer)
        lrp!(Rᵏ, rule, layer, modified_layer, aᵏ, Rᵏ⁺¹)
        @test typeof(Rᵏ) == typeof(aᵏ)
        @test size(Rᵏ) == size(aᵏ)
        @test_reference "references/rules/$rulename/$layername.jld2" Dict("R" => Rᵏ) by =
            (r, a) -> isapprox(r["R"], a["R"]; atol=1e-5, rtol=0.02)
    end
end

## Test Dense layer
# Define Dense test input
din = 4 # input dimension
dout = 3 # output dimension
batchsize = 2
aᵏ_dense = pseudorandn(din, batchsize)

layers = Dict(
    "Dense_relu"     => Dense(pseudorandn(dout, din), pseudorandn(dout), relu),
    "Dense_identity" => Dense(Matrix{Float32}(I, dout, din), false, identity),
)
@testset "Dense" begin
    for (rulename, rule) in RULES
        @testset "$rulename" begin
            for (layername, layer) in layers
                @testset "$layername" begin
                    run_rule_tests(rule, layer, rulename, layername, aᵏ_dense)
                end
            end
        end
    end
end

## Test ConvLayers and others
cin, cout = 3, 4
insize = (6, 6, 3, batchsize)
aᵏ = pseudorandn(insize)
layers = Dict(
    "Conv"           => Conv((3, 3), cin => cout; init=pseudorandn, bias=pseudorandn(cout)),
    "Conv_relu"      => Conv((3, 3), cin => cout, relu; init=pseudorandn, bias=pseudorandn(cout)),
    "MaxPool"        => MaxPool((3, 3)),
    "MeanPool"       => MeanPool((3, 3)),
    "GlobalMaxPool"  => GlobalMaxPool(),
    "GlobalMeanPool" => GlobalMeanPool(),
    "flatten"        => Flux.flatten,
    "Dropout"        => Dropout(0.2; active=false),
)
@testset "Other Layers" begin
    for (rulename, rule) in RULES
        @testset "$rulename" begin
            for (layername, layer) in layers
                @testset "$layername" begin
                    run_rule_tests(rule, layer, rulename, layername, aᵏ)
                end
            end
        end
    end
end

# Test equivalence of ZPlusRule() and AlphaBetaRule(1.0f0, 0.0f0)
layer = layers["Conv"]
Rᵏ⁺¹ = layer(aᵏ)
Rᵏ_z⁺ = similar(aᵏ)
Rᵏ_αβ = similar(aᵏ)
rule = ZPlusRule()
modified_layers = modify_layer(rule, layer)
lrp!(Rᵏ_z⁺, rule, layer, modified_layers, aᵏ, Rᵏ⁺¹)
rule = AlphaBetaRule(1.0f0, 0.0f0)
modified_layers = modify_layer(rule, layer)
lrp!(Rᵏ_αβ, rule, layer, modified_layers, aᵏ, Rᵏ⁺¹)
@test Rᵏ_z⁺ ≈ Rᵏ_αβ
