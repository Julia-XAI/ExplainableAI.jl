using LoopVectorization
using ExplainableAI
using Flux
using JLD2

const GRADIENT_ANALYZERS = Dict(
    "Gradient"            => Gradient,
    "InputTimesGradient"  => InputTimesGradient,
    "SmoothGrad"          => m -> SmoothGrad(m, 5, 0.1, MersenneTwister(123)),
    "IntegratedGradients" => m -> IntegratedGradients(m, 5),
)
const LRP_ANALYZERS = Dict(
    "LRPZero"                   => LRP,
    "LRPZero_COC"               => m -> LRP(m; flatten=false), # chain of chains
    "LRPEpsilonAlpha2Beta1Flat" => m -> LRP(m, EpsilonAlpha2Beta1Flat()),
)

input_size = (32, 32, 3, 1)
input = pseudorand(input_size)

init(dims...) = Flux.glorot_uniform(MersenneTwister(123), dims...)

model = Chain(
    Chain(
        Conv((3, 3), 3 => 8, relu; pad=1, init=init),
        Conv((3, 3), 8 => 8, relu; pad=1, init=init),
        MaxPool((2, 2)),
        Conv((3, 3), 8 => 16, relu; pad=1, init=init),
        Conv((3, 3), 16 => 16, relu; pad=1, init=init),
        MaxPool((2, 2)),
    ),
    Chain(
        Flux.flatten,
        Dense(1024 => 512, relu; init=init),         # 102_764_544 parameters
        Dropout(0.5),
        Dense(512 => 100, relu; init=init),
    ),
)
Flux.testmode!(model, true)

function test_cnn(name, method)
    @testset "$name" begin
        # Reference test attribution
        analyzer = method(model)
        print("Timing $name cold...\t")
        @time expl = analyze(input, analyzer)
        attr = expl.attribution
        @test size(attr) == size(input)
        if name == "LRPZero_COC"
            # Output of Chain of Chains should be equal to flattened model
            @test_reference "references/cnn/LRPZero.jld2" Dict("expl" => attr) by =
                (r, a) -> isapprox(r["expl"], a["expl"]; rtol=0.05)
        else
            @test_reference "references/cnn/$(name).jld2" Dict("expl" => attr) by =
                (r, a) -> isapprox(r["expl"], a["expl"]; rtol=0.05)
        end
        # Test direct call of analyzer
        analyzer = method(model)
        print("Timing $name warm...\t")
        @time expl2 = analyzer(input)
        @test expl.attribution ≈ expl2.attribution

        # Test direct call of heatmap
        h1 = heatmap(expl)
        analyzer = method(model)
        h2 = heatmap(input, analyzer)
        @test h1 ≈ h2
        if name == "LRPZero_COC"
            # Output of Chain of Chains should be equal to flattened model
            @test_reference "references/heatmaps/cnn_LRPZero.txt" h1
        elseif !in(name, ("Gradient", "SmoothGrad"))
            @test_reference "references/heatmaps/cnn_$(name).txt" h1
        end
    end
    @testset "$name neuron selection" begin
        analyzer = method(model)
        neuron_selection = 1
        expl = analyze(input, analyzer, neuron_selection)
        attr = expl.attribution

        @test size(attr) == size(input)
        if name == "LRPZero_COC"
            # Output of Chain of Chains should be equal to flattened model
            @test_reference "references/cnn/LRPZero_neuron_$neuron_selection.jld2" Dict(
                "expl" => attr
            ) by = (r, a) -> isapprox(r["expl"], a["expl"]; rtol=0.05)
        else
            @test_reference "references/cnn/$(name)_neuron_$neuron_selection.jld2" Dict(
                "expl" => attr
            ) by = (r, a) -> isapprox(r["expl"], a["expl"]; rtol=0.05)
        end
        analyzer = method(model)
        expl2 = analyzer(input, neuron_selection)
        @test expl.attribution ≈ expl2.attribution
    end
end

# Run analyzers
@testset "LRP analyzers" begin
    for (name, method) in LRP_ANALYZERS
        test_cnn(name, method)
    end
end

@testset "Gradient analyzers" begin
    for (name, method) in GRADIENT_ANALYZERS
        test_cnn(name, method)
    end
end

# Layerwise relevances in LRP methods
@testset "Layerwise relevances" begin
    analyzer1 = LRP(model)
    analyzer2 = LRP(model; flatten=false)
    e1 = analyze(input, analyzer1; layerwise_relevances=true)
    e2 = analyze(input, analyzer2; layerwise_relevances=true)
    lwr1 = e1.extras.layerwise_relevances
    lwr2 = e2.extras.layerwise_relevances

    @test length(lwr1) == 11 # 10 layers in flattened VGG11
    @test length(lwr2) == 3 # 2 chains in unflattened VGG11
    @test lwr1[1] ≈ lwr2[1]
    @test lwr1[end] ≈ lwr2[end]
end
