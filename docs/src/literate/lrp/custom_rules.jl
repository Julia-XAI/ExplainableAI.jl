# # Custom LRP rules
# One of the design goals of ExplainableAI.jl is to combine ease of use and
# extensibility for the purpose of research.
#
#
# This example will show you how to implement custom LRP rules and register custom layers
# and activation functions.
# For this purpose, we will quickly load the MNIST dataset and model from the previous section
using ExplainableAI
using Flux
using MLDatasets
using ImageCore
using BSON

index = 10
x, _ = MNIST(Float32, :test)[10]
input = reshape(x, 28, 28, 1, :)

model = BSON.load("../../model.bson", @__MODULE__)[:model]

# ## Custom LRP rules
# Let's define a rule that modifies the weights and biases of our layer on the forward pass.
# The rule has to be of type `AbstractLRPRule`.
struct MyGammaRule <: AbstractLRPRule end

# It is then possible to dispatch on the utility functions [`modify_input`](@ref ExplainableAI.modify_input),
# [`modify_parameters`](@ref ExplainableAI.modify_parameters) and [`modify_denominator`](@ref ExplainableAI.modify_denominator) with the rule type
# `MyCustomLRPRule` to define custom rules without writing any boilerplate code.
# To extend internal functions, import them explicitly:
import ExplainableAI: modify_parameters

modify_parameters(::MyGammaRule, param) = param + 0.25f0 * relu.(param)

# We can directly use this rule to make an analyzer!
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

# We just implemented our own version of the ``γ``-rule in 2 lines of code.
# The heatmap perfectly matches the previous one!

# For more granular control over weights and biases,
# [`modify_weight`](@ref ExplainableAI.modify_weight)
# and [`modify_bias`](@ref ExplainableAI.modify_bias) can be used.
# If the layer doesn't use weights `layer.weight` and biases `layer.bias`,
# ExplainableAI provides a lower-level variant of
# [`modify_parameters`](@ref ExplainableAI.modify_parameters)
# called [`modify_layer`](@ref ExplainableAI.modify_layer). This function is expected to take a layer
# and return a new, modified layer.
# To add compatibility checks between rule and layer types, extend
# [`is_compatible`](@ref ExplainableAI.is_compatible).

#md # !!! warning "Extending modify_layer"
#md #
#md #     Use of a custom function `modify_layer` will overwrite functionality of `modify_parameters`,
#md #     `modify_weight` and `modify_bias` for the implemented combination of rule and layer types.
#md #     This is due to the fact that internally, `modify_weight` and `modify_bias` are called
#md #     by the default implementation of `modify_layer`.
#md #     `modify_weight` and `modify_bias` in turn call `modify_parameters` by default.
#md #
#md #     The default call structure looks as follows:
#md #     ```
#md #     ┌─────────────────────────────────────────┐
#md #     │              modify_layer               │
#md #     └─────────┬─────────────────────┬─────────┘
#md #               │ calls               │ calls
#md #     ┌─────────▼─────────┐ ┌─────────▼─────────┐
#md #     │   modify_weight   │ │    modify_bias    │
#md #     └─────────┬─────────┘ └─────────┬─────────┘
#md #               │ calls               │ calls
#md #     ┌─────────▼─────────┐ ┌─────────▼─────────┐
#md #     │ modify_parameters │ │ modify_parameters │
#md #     └───────────────────┘ └───────────────────┘
#md #     ```
#md #
#md #     Therefore `modify_layer` should only be extended for a specific rule
#md #     and a specific layer type.

# ## Custom layers and activation functions
# ### Model checks for humans
# Good model checks and presets should allow novice users to apply XAI methods
# in a "plug & play" manner according to best practices.
#
# Let's say we define a layer that doubles its input:
struct MyDoublingLayer end
(::MyDoublingLayer)(x) = 2 * x

mylayer = MyDoublingLayer()
mylayer([1, 2, 3])

# Let's append this layer to our model:
model = Chain(model..., MyDoublingLayer())

# Creating an LRP analyzer, e.g. `LRP(model)`, will throw an `ArgumentError`
# and print a summary of the model check in the REPL:
# ```julia-repl
# ┌───┬───────────────────────┬─────────────────┬────────────┬────────────────┐
# │   │ Layer                 │ Layer supported │ Activation │ Act. supported │
# ├───┼───────────────────────┼─────────────────┼────────────┼────────────────┤
# │ 1 │ flatten               │            true │     —      │           true │
# │ 2 │ Dense(784, 100, relu) │            true │    relu    │           true │
# │ 3 │ Dense(100, 10)        │            true │  identity  │           true │
# │ 4 │ MyDoublingLayer()     │           false │     —      │           true │
# └───┴───────────────────────┴─────────────────┴────────────┴────────────────┘
#   Layers failed model check
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡
#
#   Found unknown layers MyDoublingLayer() that are not supported by ExplainableAI's LRP implementation yet.
#
#   If you think the missing layer should be supported by default, please submit an issue (https://github.com/adrhill/ExplainableAI.jl/issues).
#
#   These model checks can be skipped at your own risk by setting the LRP-analyzer keyword argument skip_checks=true.
#
#   [...]
# ```

# LRP should only be used on deep rectifier networks and ExplainableAI doesn't
# recognize `MyDoublingLayer` as a compatible layer.
# By default, it will therefore return an error and a model check summary
# instead of returning an incorrect explanation.
#
# However, if we know `MyDoublingLayer` is compatible with deep rectifier networks,
# we can register it to tell ExplainableAI that it is ok to use.
# This will be shown in the following section.

#md # !!! warning "Skipping model checks"
#md #
#md #     All model checks can be skipped at the user's own risk by setting the LRP-analyzer
#md #     keyword argument `skip_checks=true`.

# ### Registering custom layers
# The error in the model check will stop after registering our custom layer type
# `MyDoublingLayer` as "supported" by ExplainableAI.
#
# This is done using the function [`LRP_CONFIG.supports_layer`](@ref),
# which should be set to return `true` for the type `MyDoublingLayer`:
LRP_CONFIG.supports_layer(::MyDoublingLayer) = true

# Now we can create and run an analyzer without getting an error:
analyzer = LRP(model)
heatmap(input, analyzer)

#md # !!! note "Registering functions"
#md #
#md #     Flux's `Chains` can also contain functions, e.g. `flatten`.
#md #     This kind of layer can be registered as
#md #     ```julia
#md #     LRP_CONFIG.supports_layer(::typeof(mylayer)) = true
#md #     ```

# ### Registering activation functions
# The mechanism for registering custom activation functions is analogous to that of custom layers:
myrelu(x) = max.(0, x)
model = Chain(Flux.flatten, Dense(784, 100, myrelu), Dense(100, 10))

# Once again, creating an LRP analyzer for this model will throw an `ArgumentError`
# and display the following model check summary:
# ```julia-repl
# julia> analyzer = LRP(model3)
# ┌───┬─────────────────────────┬─────────────────┬────────────┬────────────────┐
# │   │ Layer                   │ Layer supported │ Activation │ Act. supported │
# ├───┼─────────────────────────┼─────────────────┼────────────┼────────────────┤
# │ 1 │ flatten                 │            true │     —      │           true │
# │ 2 │ Dense(784, 100, myrelu) │            true │   myrelu   │          false │
# │ 3 │ Dense(100, 10)          │            true │  identity  │           true │
# └───┴─────────────────────────┴─────────────────┴────────────┴────────────────┘
#   Activations failed model check
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡
#
#   Found layers with unknown or unsupported activation functions myrelu. LRP assumes that the model is a "deep rectifier network" that only contains ReLU-like activation functions.
#
#   If you think the missing activation function should be supported by default, please submit an issue (https://github.com/adrhill/ExplainableAI.jl/issues).
#
#   These model checks can be skipped at your own risk by setting the LRP-analyzer keyword argument skip_checks=true.
#
#   [...]
# ```

# Registation works by defining the function [`LRP_CONFIG.supports_activation`](@ref) as `true`:
LRP_CONFIG.supports_activation(::typeof(myrelu)) = true

# now the analyzer can be created without error:
analyzer = LRP(model)
