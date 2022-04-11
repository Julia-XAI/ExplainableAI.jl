using ExplainableAI
using Flux
using MLDatasets
using ImageCore
using BSON

model = BSON.load("../model.bson", @__MODULE__)[:model]

index = 10
x, y = MNIST.testdata(Float32, index)
input = reshape(x, 28, 28, 1, :);

rules = [
    ZBoxRule(),
    GammaRule(),
    GammaRule(),
    EpsilonRule(),
    EpsilonRule(),
    EpsilonRule(),
    ZeroRule(),
    ZeroRule(),
]

analyzer = LRP(model, rules)
heatmap(input, analyzer)

struct MyGammaRule <: AbstractLRPRule end

import ExplainableAI: modify_params

function modify_params(::MyGammaRule, W, b)
    ρW = W + 0.25 * relu.(W)
    ρb = b + 0.25 * relu.(b)
    return ρW, ρb
end

analyzer = LRP(model, MyGammaRule())
heatmap(input, analyzer)

analyzer = LRP(model, GammaRule())
heatmap(input, analyzer)

struct MyDoublingLayer end
(::MyDoublingLayer)(x) = 2 * x

mylayer = MyDoublingLayer()
mylayer([1, 2, 3])

model = Chain(model..., MyDoublingLayer())

LRP_CONFIG.supports_layer(::MyDoublingLayer) = true

analyzer = LRPZero(model)
heatmap(input, analyzer)

myrelu(x) = max.(0, x)
model = Chain(Flux.flatten, Dense(784, 100, myrelu), Dense(100, 10))

LRP_CONFIG.supports_activation(::typeof(myrelu)) = true

analyzer = LRPZero(model)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

