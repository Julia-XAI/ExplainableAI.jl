# # Custom LRP rules
# One of the design goals of ExplainableAI.jl is to combine ease of use and
# extensibility for the purpose of research.
# This example will show you how to implement custom LRP rules.
#
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

# ## Step 1: Define struct of supertype `AbstractLRPRule`
# Let's define a rule that modifies the weights and biases of our layer on the forward pass.
# The rule has to be of supertype `AbstractLRPRule`.
struct MyGammaRule <: AbstractLRPRule end

# ## Step 2: implement 4 functions
# It is then possible to dispatch on the following four utility functions
# with the rule type `MyCustomLRPRule` to define custom rules without writing boilerplate code.
#
# 1. [`modify_input(rule, input)`](@ref ExplainableAI.modify_input)
# 1. [`modify_parameters(rule, parameter)`](@ref ExplainableAI.modify_parameters)
# 1. [`modify_denominator(rule, denominator)`](@ref ExplainableAI.modify_denominator)
# 1. [`is_compatible(rule, layer)`](@ref ExplainableAI.is_compatible)
#
# By default:
# 1. `modify_input` doesn't change the input
# 1. `modify_parameters` doesn't change the parameters
# 1. `modify_denominator` avoids division by zero by adding a small epsilon-term (`1.0f-9`)
# 1. `is_compatible` returns `true` if a layer has fields `weight` and `bias`
#
# To extend internal functions, import them explicitly:
import ExplainableAI: modify_parameters

modify_parameters(::MyGammaRule, param) = param + 0.25f0 * relu.(param)

# Note that we didn't implement three of the four functions.
# This is because the defaults are sufficient for the `GammaRule`.

# ## Step 3: Use your rule
# We can directly use this rule to make an analyzer!
rules = [
    FlatRule(),
    EpsilonRule(),
    MyGammaRule(), # our custom GammaRule
    EpsilonRule(),
    ZeroRule(),
    ZeroRule(),
    ZeroRule(),
    ZeroRule(),
]
analyzer = LRP(model, rules)
heatmap(input, analyzer)

# We just implemented our own version of the ``γ``-rule in 2 lines of code.
# The heatmap perfectly matches the pre-implemented `GammaRule`:
rules = [
    FlatRule(),
    EpsilonRule(),
    GammaRule(), # XAI.jl's GammaRule
    EpsilonRule(),
    ZeroRule(),
    ZeroRule(),
    ZeroRule(),
    ZeroRule(),
]
analyzer = LRP(model, rules)
heatmap(input, analyzer)

# ## Advanced layer modification
# For more granular control over weights and biases,
# [`modify_weight`](@ref ExplainableAI.modify_weight) and
# [`modify_bias`](@ref ExplainableAI.modify_bias) can be used.
#
# If the layer doesn't use weights (`layer.weight`) and biases (`layer.bias`),
# ExplainableAI provides a lower-level variant of
# [`modify_parameters`](@ref ExplainableAI.modify_parameters) called
# [`modify_layer`](@ref ExplainableAI.modify_layer).
# This function is expected to take a layer and return a new, modified layer.
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

# ## Advanced LRP rules
# To implement custom LRP rules that require more than `modify_layer`, `modify_input`
# and `modify_denominator`, take a look at the [LRP developer documentation](@ref lrp-dev-docs).

# ## Performance tips
# 1. Make sure functions like `modify_parameters` don't promote the type of weights
#    (e.g. from `Float32` to `Float64`).
# 2. If your rule `MyRule` doesn't modify weights or biases,
#    defining `modify_layer(::MyRule, layer) = nothing`
#    can provide reduce memory allocations and improve performance.
