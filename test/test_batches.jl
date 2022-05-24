using Flux
using ExplainableAI
using Random
using Distributions: Laplace

## Test `fuse_batchnorm` on Dense and Conv layers
ins = 20
outs = 10
batchsize = 15

pseudorand(n...) = rand(MersenneTwister(123), Float32, n...)
model = Chain(Dense(ins, outs, relu; init=pseudorand))

# Input 1 w/o batch dimension
input1_no_bd = rand(MersenneTwister(1), Float32, ins)
# Input 1 with batch dimension
input1_bd = reshape(input1_no_bd, ins, 1)
# Input 2 with batch dimension
input2_bd = rand(MersenneTwister(2), Float32, ins, 1)
# Batch containing inputs 1 & 2
input_batch = cat(input1_bd, input2_bd; dims=2)

ANALYZERS = Dict(
    "LRPZero" => LRPZero,
    "Gradient" => Gradient,
    "InputTimesGradient" => InputTimesGradient,
    "SmoothGrad" => m -> SmoothGrad(m, 5, 0.1, MersenneTwister(123)),
    "SmoothLRP" =>
        m -> NoiseAugmentation(LRP(m), 2, Laplace(0.0f0, 0.1f0), MersenneTwister(123)),
    "IntegratedGradients" => m -> IntegratedGradients(m, 5),
    "IntegratedLRP" => m -> IntegrationAugmentation(LRP(m), 2),
)

for (name, method) in ANALYZERS
    @testset "$name" begin
        # Using `add_batch_dim=true` should result in same attribution
        # as input reshaped to have a batch dimension
        analyzer = method(model)
        expl1_no_bd = analyzer(input1_no_bd; add_batch_dim=true)
        analyzer = method(model)
        expl1_bd = analyzer(input1_bd)
        @test expl1_bd.attribution ≈ expl1_no_bd.attribution

        # Analyzing a batch should have the same result
        # as analyzing inputs in batch individually
        analyzer = method(model)
        expl2_bd = analyzer(input2_bd)
        analyzer = method(model)
        expl_batch = analyzer(input_batch)
        @test expl1_bd.attribution ≈ expl_batch.attribution[:, 1]
        if !(analyzer isa NoiseAugmentation)
            # NoiseAugmentation methods generate random numbers for the entire batch.
            # therefore explanations don't match except for the first input in the batch.
            @test expl2_bd.attribution ≈ expl_batch.attribution[:, 2]
        end
    end
end
