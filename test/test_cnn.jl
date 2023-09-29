using LoopVectorization
using ExplainableAI
using Flux
using JLD2

const GRADIENT_ANALYZERS = Dict(
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
        Dense(1024 => 512, relu; init=init),
        Dropout(0.5),
        Dense(512 => 100, relu; init=init),
    ),
)
Flux.testmode!(model, true)

@testset "Test API" begin
    analyzer = LRP(model)
    println("Timing Gradient...")
    print("cold:")
    @time expl = analyze(input, analyzer)

    # Test direct call of analyzer
    print("warm:")
    @time expl2 = analyzer(input)
    @test expl.val ≈ expl2.val

    @test_reference "references/cnn/Gradient_max.jld2" Dict("expl" => expl.val) by =
        (r, a) -> isapprox(r["expl"], a["expl"]; rtol=0.05)

    # Test direct call of heatmap
    h1 = heatmap(expl)
    h2 = heatmap(input, analyzer)
    @test h1 ≈ h2
    @test_reference "references/heatmaps/cnn_Gradient.txt" h1

    # Test neuron selection
    expl = analyze(input, analyzer, 1)
    expl2 = analyzer(input, 1)
    @test expl.val ≈ expl2.val
    @test_reference "references/cnn/Gradient_ns1.jld2" Dict("expl" => expl.val) by =
        (r, a) -> isapprox(r["expl"], a["expl"]; rtol=0.05)
end

function test_cnn(name, method)
    @testset "$name" begin
        @testset "Max activation" begin
            # Reference test explanation
            analyzer = method(model)
            println("Timing $name...")
            print("cold:")
            @time expl = analyze(input, analyzer)

            @test size(expl.val) == size(input)
            @test_reference "references/cnn/$(name)_max.jld2" Dict("expl" => expl.val) by =
                (r, a) -> isapprox(r["expl"], a["expl"]; rtol=0.05)
        end
        @testset "Neuron selection" begin
            analyzer = method(model)
            print("warm:")
            @time expl = analyze(input, analyzer, 1)

            @test size(expl.val) == size(input)
            @test_reference "references/cnn/$(name)_ns1.jld2" Dict("expl" => expl.val) by =
                (r, a) -> isapprox(r["expl"], a["expl"]; rtol=0.05)
        end
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

@testset "CRP" begin
    composite = EpsilonPlus()
    layer_index = 5 # last Conv layer
    n_concepts = 2
    concepts = TopNConcepts(n_concepts)
    analyzer = CRP(LRP(model, composite), layer_index, concepts)

    @testset "Max activation" begin
        println("Timing CRP...")
        print("cold:")
        @time expl = analyze(input, analyzer)

        @test size(expl.val) == size(input) .* (1, 1, 1, n_concepts)
        @test_reference "references/cnn/CRP_max.jld2" Dict("expl" => expl.val) by =
            (r, a) -> isapprox(r["expl"], a["expl"]; rtol=0.05)
    end
    @testset "Neuron selection" begin
        print("warm:")
        @time expl = analyze(input, analyzer, 1)

        @test size(expl.val) == size(input) .* (1, 1, 1, n_concepts)
        @test_reference "references/cnn/CRP_ns1.jld2" Dict("expl" => expl.val) by =
            (r, a) -> isapprox(r["expl"], a["expl"]; rtol=0.05)
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
