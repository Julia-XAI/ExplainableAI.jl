using ExplainableAI
using Flux
using MLDatasets
using ImageCore
using BSON

model = BSON.load("../model.bson", @__MODULE__)[:model]

index = 10
x, _ = MNIST(Float32, :test)[10]
input = reshape(x, 28, 28, 1, :);

rules = [
    ZBoxRule(0.0f0, 1.0f0),
    EpsilonRule(),
    GammaRule(),
    EpsilonRule(),
    ZeroRule(),
    ZeroRule(),
    ZeroRule(),
    ZeroRule(),
]

analyzer = LRP(model, rules)

heatmap(input, analyzer)

composite = Composite(
    ZeroRule(),                              # default rule
    GlobalTypeMap(
        Conv => GammaRule(),                 # apply GammaRule on all convolutional layers
        MaxPool => EpsilonRule(),            # apply EpsilonRule on all pooling-layers
    ),
    FirstLayerMap(ZBoxRule(0.0f0, 1.0f0)),   # apply ZBoxRule on the first layer
)

analyzer = LRP(model, composite)

heatmap(input, analyzer)

composite = EpsilonPlusFlat()

analyzer = LRP(model, composite)

struct MyGammaRule <: AbstractLRPRule end

import ExplainableAI: modify_parameters

modify_parameters(::MyGammaRule, param) = param + 0.25f0 * relu.(param)

rules = [
    ZBoxRule(0.0f0, 1.0f0),
    EpsilonRule(),
    MyGammaRule(),
    EpsilonRule(),
    ZeroRule(),
    ZeroRule(),
    ZeroRule(),
    ZeroRule(),
]
analyzer = LRP(model, rules)
heatmap(input, analyzer)

struct MyDoublingLayer end
(::MyDoublingLayer)(x) = 2 * x

mylayer = MyDoublingLayer()
mylayer([1, 2, 3])

model = Chain(model..., MyDoublingLayer())

LRP_CONFIG.supports_layer(::MyDoublingLayer) = true

analyzer = LRP(model)
heatmap(input, analyzer)

myrelu(x) = max.(0, x)
model = Chain(Flux.flatten, Dense(784, 100, myrelu), Dense(100, 10))

LRP_CONFIG.supports_activation(::typeof(myrelu)) = true

analyzer = LRP(model)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

