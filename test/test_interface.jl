using ExplainableAI
using Zygote
using Test

using Flux
using StableRNGs: StableRNG

pseudorand(dims...) = rand(StableRNG(123), Float32, dims...)

model = Chain(
    Chain(Conv((3, 3), 3 => 4, relu), MaxPool((2, 2))),
    Chain(Flux.flatten, Dense(36 => 5, relu), Dense(5 => 3)),
)
input = pseudorand(8, 8, 3, 2)

ANALYZERS = Dict(
    "Gradient" => (Gradient(model), NormPooling),
    "InputTimesGradient" => (InputTimesGradient(model), SumPooling),
    "SmoothGrad" => (SmoothGrad(model, 5, 0.1f0, StableRNG(123), false), NormPooling),
    "IntegratedGradients" => (IntegratedGradients(model, 5), SumPooling),
    "GradCAM" => (GradCAM(model[1], model[2]), UnsignedNoPooling),
    "NoiseAugmentation" => (
        NoiseAugmentation(
            InputTimesGradient(model), 5, 0.1f0, StableRNG(123), false;
            pooling = SumAbsPooling(),
        ),
        SumAbsPooling,
    ),
    "InterpolationAugmentation" => (
        InterpolationAugmentation(Gradient(model), 5; pooling = MaxPooling()),
        MaxPooling,
    ),
)

@testset "$name" for (name, (analyzer, P)) in ANALYZERS
    @test XAIBase.test_interface(analyzer, input)
    @test XAIBase.test_interface(analyzer, input; output_selection = 2)
    @test analyze(input, analyzer).pooling isa P
end
