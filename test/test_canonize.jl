using Flux
using ExplainableAI
using ExplainableAI: fuse_batchnorm
using Random

pseudorand(dims...) = rand(MersenneTwister(123), Float32, dims...)
batchsize = 1000

# # Test `fuse_batchnorm` on Dense layer
ins = 20
outs = 10
dense = Dense(ins, outs; init=pseudorand)
bn_dense = BatchNorm(outs, relu; initβ=pseudorand, initγ=pseudorand)
model = Chain(dense, bn_dense)

# collect statistics
x = pseudorand(ins, batchsize)
Flux.trainmode!(model)
model(x)
Flux.testmode!(model)

dense_fused = @inferred fuse_batchnorm(dense, bn_dense)
@test dense_fused(x) ≈ model(x)

# # Test `fuse_batchnorm` on Conv layer
insize = (10, 10, 3)
conv = Conv((3, 3), 3 => 4; init=pseudorand)
bn_conv = BatchNorm(4, relu; initβ=pseudorand, initγ=pseudorand)
model = Chain(conv, bn_conv)

# collect statistics
x = pseudorand(insize..., batchsize)
Flux.trainmode!(model)
model(x)
Flux.testmode!(model)

conv_fused = @inferred fuse_batchnorm(conv, bn_conv)
@test conv_fused(x) ≈ model(x)

# # Test `canonize` on models
# Sequential BatchNorm layers should be fused until they create a Dense or Conv layer
# with non-linear activation function.
model = Chain(
    Conv((3, 3), 3 => 6),
    BatchNorm(6),
    Conv((3, 3), 6 => 2, identity),
    BatchNorm(2),
    BatchNorm(2, softplus),
    BatchNorm(2),
    flatten,
    Dense(72, 10),
    BatchNorm(10),
    BatchNorm(10),
    BatchNorm(10, relu),
    BatchNorm(10),
    Dense(10, 10, gelu),
    BatchNorm(10),
    softmax,
)
Flux.trainmode!(model)
model(x)
Flux.testmode!(model)
model_canonized = canonize(model)

# 6 of the BatchNorm layers should be removed and the ouputs should match
@test length(model_canonized) == length(model) - 6
@test model(x) ≈ model_canonized(x)
