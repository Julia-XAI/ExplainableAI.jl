using ExplainableAI
using Flux
using JLD2

const GRADIENT_ANALYZERS = Dict(
    "Gradient" => Gradient, "InputTimesGradient" => InputTimesGradient
)
const LRP_ANALYZERS = Dict(
    "LRPZero" => LRPZero, "LRPEpsilon" => LRPEpsilon, "LRPGamma" => LRPGamma
)

using Random
pseudorand(T, dims...) = rand(MersenneTwister(123), T, dims...)
input_size = (224, 224, 3, 1)
img = pseudorand(Float32, input_size)

# Load VGG model:
# We run the reference test on the randomly intialized weights
# so we don't have to download ~550 MB on every CI run.
include("./vgg11.jl")
vgg11 = VGG11(; pretrain=false)
model = flatten_model(strip_softmax(vgg11.layers))

function LRPCustom(model::Chain)
    return LRP(model, [ZBoxRule(), repeat([GammaRule()], length(model.layers) - 1)...])
end

function test_vgg11(name, method; kwargs...)
    analyzer = method(model)
    @testset "$name" begin
        # Reference test attribution
        print("Timing $name...\t")
        @time expl = analyze(img, analyzer; kwargs...)
        attr = expl.attribution
        @test size(attr) == size(img)
        @test_reference "references/vgg11/$(name).jld2" Dict("expl" => attr) by =
            (r, a) -> isapprox(r["expl"], a["expl"]; rtol=0.05)

        # Test direct call of analyzer
        expl2 = analyzer(img; kwargs...)
        @test expl.attribution ≈ expl2.attribution

        # Test direct call of heatmap
        h1 = heatmap(expl)
        h2 = heatmap(img, analyzer; kwargs...)
        @test h1 ≈ h2
        if name != "Gradient" # TODO: remove
            @test_reference "references/heatmaps/vgg11_$(name).txt" h1
        end
    end
    @testset "$name neuron selection" begin
        neuron_selection = 1
        expl = analyze(img, analyzer, neuron_selection; kwargs...)
        attr = expl.attribution

        @test size(attr) == size(img)
        @test_reference "references/vgg11/$(name)_neuron_$neuron_selection.jld2" Dict(
            "expl" => attr
        ) by = (r, a) -> isapprox(r["expl"], a["expl"]; rtol=0.05)

        expl2 = analyzer(img, neuron_selection; kwargs...)
        @test expl.attribution ≈ expl2.attribution
    end
    return nothing
end

# Run analyzers
@testset "LRP analyzers" begin
    for (name, method) in LRP_ANALYZERS
        test_vgg11(name, method)
    end
end
@testset "Custom LRP composite" begin
    test_vgg11("LRPCustom", LRPCustom)
end

@testset "Gradient analyzers" begin
    for (name, method) in GRADIENT_ANALYZERS
        test_vgg11(name, method)
    end
end
# Layerwise relevances in LRP methods
@testset "Layerwise relevances" begin
    test_vgg11("LRPZero", LRPZero; layerwise_relevances=true)
end

# Test LRP constructor with no rules
a1 = LRP(model)
a2 = LRPZero(model)
@test a1.model == a2.model
@test a1.rules == a2.rules
