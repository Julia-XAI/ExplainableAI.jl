using Flux
using ExplainableAI

struct MyDoublingLayer end
(::MyDoublingLayer)(x) = 2 * x

mylayer = MyDoublingLayer()
mylayer([1, 2, 3])

model = Chain(
    Dense(100, 20),
    MyDoublingLayer()
);

LRP_CONFIG.supports_layer(::MyDoublingLayer) = true

analyzer = LRP(model)

myrelu(x) = max.(0, x)

model = Chain(
    Dense(784, 100, myrelu),
    Dense(100, 10),
);

LRP_CONFIG.supports_activation(::typeof(myrelu)) = true

analyzer = LRP(model)

struct UnknownLayer end
(::UnknownLayer)(x) = x

unknown_activation(x) = max.(0, x)

model = Chain(Dense(100, 20, unknown_activation), MyDoublingLayer())

LRP(model; skip_checks=true)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
