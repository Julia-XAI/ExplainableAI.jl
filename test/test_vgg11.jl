using LoopVectorization
using ExplainableAI
using Flux
using JLD2

const GRADIENT_ANALYZERS = Dict(
    "Gradient"            => Gradient,
    "InputTimesGradient"  => InputTimesGradient,
    "SmoothGrad"          => model -> SmoothGrad(model, 5, 0.1, MersenneTwister(123)),
    "IntegratedGradients" => model -> IntegratedGradients(model, 5),
)
const LRP_ANALYZERS = Dict(
    "LRPZero" => LRP,
    "LRPEpsilonAlpha2Beta1Flat" => model -> LRP(model, EpsilonAlpha2Beta1Flat()),
)

input_size = (224, 224, 3, 1)
img = pseudorand(input_size)

model = strip_softmax(vgg11.layers)
Flux.testmode!(model, true)

# TODO: add test comparing flattened and unflattened model

function test_vgg11(name, method; kwargs...)
    @testset "$name" begin
        # Reference test attribution
        analyzer = method(model)
        print("Timing $name cold...\t")
        @time expl = analyze(img, analyzer; kwargs...)
        attr = expl.attribution
        @test size(attr) == size(img)
        @test_reference "references/vgg11/$(name).jld2" Dict("expl" => attr) by =
            (r, a) -> isapprox(r["expl"], a["expl"]; rtol=0.05)

        # Test direct call of analyzer
        analyzer = method(model)
        print("Timing $name warm...\t")
        @time expl2 = analyzer(img; kwargs...)
        @test expl.attribution ≈ expl2.attribution

        # Test direct call of heatmap
        h1 = heatmap(expl)
        analyzer = method(model)
        h2 = heatmap(img, analyzer; kwargs...)
        @test h1 ≈ h2
        if !in(name, ("Gradient", "SmoothGrad"))
            @test_reference "references/heatmaps/vgg11_$(name).txt" h1
        end
    end
    @testset "$name neuron selection" begin
        analyzer = method(model)
        neuron_selection = 1
        expl = analyze(img, analyzer, neuron_selection; kwargs...)
        attr = expl.attribution

        @test size(attr) == size(img)
        @test_reference "references/vgg11/$(name)_neuron_$neuron_selection.jld2" Dict(
            "expl" => attr
        ) by = (r, a) -> isapprox(r["expl"], a["expl"]; rtol=0.05)
        analyzer = method(model)
        expl2 = analyzer(img, neuron_selection; kwargs...)
        @test expl.attribution ≈ expl2.attribution
    end
end

# Run analyzers
@testset "LRP analyzers" begin
    for (name, method) in LRP_ANALYZERS
        test_vgg11(name, method)
    end
end

@testset "Gradient analyzers" begin
    for (name, method) in GRADIENT_ANALYZERS
        test_vgg11(name, method)
    end
end

# Layerwise relevances in LRP methods
@testset "Layerwise relevances" begin
    test_vgg11("LRPZero", LRP; layerwise_relevances=true)
end
