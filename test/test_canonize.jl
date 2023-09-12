using Flux
using Flux: flatten
using ExplainableAI
using ExplainableAI: fuse_batchnorm
using Random

batchsize = 50

#======================================#
# Test `fuse_batchnorm` on Dense layer #
#======================================#

ins = 10
outs = 5
dense = Dense(ins, outs; init=pseudorand)
bn_dense = BatchNorm(outs, relu; initβ=pseudorand, initγ=pseudorand)
model = Chain(dense, bn_dense)

# collect statistics
x = pseudorand(ins, batchsize)
Flux.testmode!(model, false)
model(x)
Flux.testmode!(model, true)

dense_fused = @inferred fuse_batchnorm(dense, bn_dense)
@test dense_fused(x) ≈ model(x)

#=====================================#
# Test `fuse_batchnorm` on Conv layer #
#=====================================#

insize = (10, 10, 3)
conv = Conv((3, 3), 3 => 4; init=pseudorand)
bn_conv = BatchNorm(4, relu; initβ=pseudorand, initγ=pseudorand)
model = Chain(conv, bn_conv)

# collect statistics
x = pseudorand(insize..., batchsize)
Flux.testmode!(model, false)
model(x)
Flux.testmode!(model, true)

conv_fused = @inferred fuse_batchnorm(conv, bn_conv)
@test conv_fused(x) ≈ model(x)

#======================================#
# Test `canonize` on sequential models #
#======================================#

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
    Dense(72, 10; bias=false),
    BatchNorm(10),
    BatchNorm(10),
    BatchNorm(10, relu),
    BatchNorm(10),
    Dense(10, 10, gelu),
    BatchNorm(10),
    softmax,
)

# collect statistics
Flux.testmode!(model, false)
model(x)
Flux.testmode!(model, true)
model_canonized = canonize(model)

# 6 of the BatchNorm layers should be removed and the ouputs should match
@test length(model_canonized) == 9 # 15 - 6
@test model(x) ≈ model_canonized(x)

#====================================#
# Test `canonize` on nested models   #
#====================================#

model = Chain(
    Conv((3, 3), 3 => 4),
    BatchNorm(4, relu),
    Chain(
        Conv((3, 3), 4 => 5, identity),
        BatchNorm(5),
        Chain(Conv((3, 3), 5 => 6, identity), BatchNorm(6)),
        Chain(Conv((3, 3), 6 => 7, identity), BatchNorm(7)),
    ),
)

# collect statistics
Flux.testmode!(model, false)
model(x)
Flux.testmode!(model, true)
model_canonized = canonize(model)

# 6 of the BatchNorm layers should be removed and the ouputs should match
@test length(model_canonized) == 4
@test model(x) ≈ model_canonized(x)

#=================================================#
# Test `canonize` on  models w/ Parallel layers   #
#=================================================#

model = Chain(
    Conv((3, 3), 3 => 4), # fuse
    BatchNorm(4, relu),
    Parallel(
        +,
        Conv((3, 3), 4 => 5, identity),
        Chain(
            Conv((3, 3), 4 => 5; bias=false, identity), # fuse
            BatchNorm(5),
        ),
        Chain(
            Conv((3, 3), 4 => 5, relu), # don't fuse
            BatchNorm(5),
        ),
    ),
    Conv((3, 3), 5 => 6, identity), # fuse
    BatchNorm(6),
)

# collect statistics
Flux.testmode!(model, false)
model(x)
Flux.testmode!(model, true)
model_canonized = canonize(model)

@test length(model_canonized) == 3
@test length(model_canonized[2][2]) == 1
@test length(model_canonized[2][3]) == 2
@test model(x) ≈ model_canonized(x)
