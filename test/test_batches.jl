using ExplainableAI
using Zygote
using Test

using Flux
using Random
using StableRNGs: StableRNG
using Distributions: Laplace

pseudorand(dims...) = rand(StableRNG(123), Float32, dims...)

## Test `fuse_batchnorm` on Dense and Conv layers
ins = 20
outs = 10
batchsize = 15

model = Chain(Dense(ins, 15, relu; init = pseudorand), Dense(15, outs, relu; init = pseudorand))

# Input 1 with batch dimension
input1 = rand(StableRNG(1), Float32, ins, 1)
# Input 2 with batch dimension
input2 = rand(StableRNG(2), Float32, ins, 1)
# Batch containing inputs 1 & 2
input_batch = cat(input1, input2; dims = 2)

ANALYZERS = Dict(
    "Gradient" => Gradient,
    "InputTimesGradient" => InputTimesGradient,
    "SmoothGrad" => m -> SmoothGrad(m, 5, 0.1, StableRNG(123)),
    "IntegratedGradients" => m -> IntegratedGradients(m, 5),
    "GradCAM" => m -> GradCAM(m[1], m[2]),
)

for (name, method) in ANALYZERS
    @testset "$name" begin
        analyzer = method(model)
        expl1 = analyzer(input1)
        @test expl1.val ≈ expl1.val

        # Analyzing a batch should have the same result
        # as analyzing inputs in batch individually
        analyzer = method(model)
        expl2 = analyzer(input2)
        analyzer = method(model)
        expl_batch = analyzer(input_batch)
        @test expl1.val ≈ expl_batch.val[:, 1]
        if !(analyzer isa NoiseAugmentation)
            # NoiseAugmentation methods generate random numbers for the entire batch.
            # therefore explanations don't match except for the first input in the batch.
            @test expl2.val ≈ expl_batch.val[:, 2]
        end
    end
end
