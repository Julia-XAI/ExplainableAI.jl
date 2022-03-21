# # Advanced LRP usage
# One of the design goals of ExplainabilityMethods.jl is to combine ease of use and
# extensibility for the purpose of research.
#
#
# This example will show you how to implement custom LRP rules and register custom layers
# and activation functions.
# For this purpose, we will quickly load our model from the previous section:
using ExplainabilityMethods
using Flux
using MLDatasets
using ImageCore
using BSON

model = BSON.load("../model.bson", @__MODULE__)[:model]

index = 10
x, y = MNIST.testdata(Float32, index)
input = reshape(x, 28, 28, 1, :);

# ## Custom LRP composites
# Instead of creating an LRP-analyzer from a single rule (e.g. `LRP(model, GammaRule())`),
# we can also assign rules to each layer individually.
# For this purpose, we create an array of rules that matches the length of the Flux chain:
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

# Since some Flux Chains contain other Flux Chains, ExplainabilityMethods provides
# a utility function called [`flatten_model`](@ref).
#
#md # !!! warning "Flattening models"
#md #     Not all models can be flattened, e.g. those using
#md #     `Parallel` and `SkipConnection` layers.

# ## Custom LRP rules
# Let's define a rule that modifies the weights and biases of our layer on the forward pass.
# The rule has to be of type `AbstractLRPRule`.
struct MyGammaRule <: AbstractLRPRule end

# It is then possible to dispatch on the utility functions  [`modify_params`](@ref) and [`modify_denominator`](@ref)
# with the rule type `MyCustomLRPRule` to define custom rules without writing any boilerplate code.
# To extend internal functions, import them explicitly:
import ExplainabilityMethods: modify_params

function modify_params(::MyGammaRule, W, b)
    ρW = W + 0.25 * relu.(W)
    ρb = b + 0.25 * relu.(b)
    return ρW, ρb
end

# We can directly use this rule to make an analyzer!
analyzer = LRP(model, MyGammaRule())
heatmap(input, analyzer)

# We just implemented our own version of the ``γ``-rule in 7 lines of code!
# The outputs match perfectly:
analyzer = LRP(model, GammaRule())
heatmap(input, analyzer)

# If the layer doesn't use weights and biases `W` and `b`, ExplainabilityMethods provides a
# lower-level variant of [`modify_params`](@ref) called [`modify_layer`](@ref).
# This function is expected to take a layer and return a new, modified layer.

#md # !!! warning "Using modify_layer"
#md #
#md #     Use of the function `modify_layer` will overwrite functionality of `modify_params`
#md #     for the implemented combination of rule and layer types.
#md #     This is due to the fact that internally, `modify_params` is called by the default
#md #     implementation of `modify_layer`.
#md #
#md #     Therefore it is recommended to only extend `modify_layer` for a specific rule
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

# Creating an LRP analyzer, e.g. `LRPZero(model)`, will throw an `ArgumentError`
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
#   Found unknown layers MyDoublingLayer() that are not supported by ExplainabilityMethods' LRP implementation yet.
#
#   If you think the missing layer should be supported by default, please submit an issue (https://github.com/adrhill/ExplainabilityMethods.jl/issues).
#
#   These model checks can be skipped at your own risk by setting the LRP-analyzer keyword argument skip_checks=true.
#
#   [...]
# ```

# LRP should only be used on deep rectifier networks and ExplainabilityMethods doesn't
# recognize `MyDoublingLayer` as a compatible layer.
# By default, it will therefore return an error and a model check summary
# instead of returning an incorrect explanation.
#
# However, if we know `MyDoublingLayer` is compatible with deep rectifier networks,
# we can register it to tell ExplainabilityMethods that it is ok to use.
# This will be shown in the following section.

#md # !!! warning "Skipping model checks"
#md #
#md #     All model checks can be skipped at the user's own risk by setting the LRP-analyzer
#md #     keyword argument `skip_checks=true`.

# ### Registering custom layers
# The error in the model check will stop after registering our custom layer type
# `MyDoublingLayer` as "supported" by ExplainabilityMethods.
#
# This is done using the function [`LRP_CONFIG.supports_layer`](@ref),
# which should be set to return `true` for the type `MyDoublingLayer`:
LRP_CONFIG.supports_layer(::MyDoublingLayer) = true

# Now we can create and run an analyzer without getting an error:
analyzer = LRPZero(model)
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
model = Chain(flatten, Dense(784, 100, myrelu), Dense(100, 10))

# Once again, creating an LRP analyzer for this model will throw an `ArgumentError`
# and display the following model check summary:
# ```julia-repl
# julia> analyzer = LRPZero(model3)
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
#   If you think the missing activation function should be supported by default, please submit an issue (https://github.com/adrhill/ExplainabilityMethods.jl/issues).
#
#   These model checks can be skipped at your own risk by setting the LRP-analyzer keyword argument skip_checks=true.
#
#   [...]
# ```

# Registation works by defining the function [`LRP_CONFIG.supports_activation`](@ref) as `true`:
LRP_CONFIG.supports_activation(::typeof(myrelu)) = true

# now the analyzer can be created without error:
analyzer = LRPZero(model)

# ## How it works internally
# Internally, ExplainabilityMethods dispatches to low level functions
# ```julia
# function lrp!(rule, layer, Rₖ, aₖ, Rₖ₊₁)
#     Rₖ .= ...
# end
# ```
# These functions use the arguments `rule` and `layer` to dispatch
# `modify_params` and `modify_denominator` on the rule and layer type.
# They in-place modify a pre-allocated array of the input relevance `Rₖ`
# based on the input activation `aₖ` and output relevance `Rₖ₊₁`.
#
# Calling `analyze` then applies a foward-pass of the model, keeping track of
# the activations `aₖ` for each layer `k`.
# The relevance `Rₖ₊₁` is then set to the output neuron activation and the rules are applied
# in a backward-pass over the model layers and previous activations.

# ### Generic rule implementation using automatic differentiation
# The generic LRP rule–of which the ``0``-, ``\epsilon``- and ``\gamma``-rules are special cases–reads[^1][^2]:
# ```math
# R_{j}=\sum_{k} \frac{a_{j} \cdot \rho\left(w_{j k}\right)}{\epsilon+\sum_{0, j} a_{j} \cdot \rho\left(w_{j k}\right)} R_{k}
# ```
#
# where ``\rho`` is a function that modifies parameters – what we have so far called `modify_params`.
#
# The computation of this propagation rule can be decomposed into four steps:
# ```math
# \begin{array}{lr}
# \forall_{k}: z_{k}=\epsilon+\sum_{0, j} a_{j} \cdot \rho\left(w_{j k}\right) & \text { (forward pass) } \\
# \forall_{k}: s_{k}=R_{k} / z_{k} & \text { (element-wise division) } \\
# \forall_{j}: c_{j}=\sum_{k} \rho\left(w_{j k}\right) \cdot s_{k} & \text { (backward pass) } \\
# \forall_{j}: R_{j}=a_{j} c_{j} & \text { (element-wise product) }
# \end{array}
# ```
#
# For deep rectifier networks, the third step can also be written as the gradient computation
# ```math
# c_{j}=\left[\nabla\left(\sum_{k} z_{k}(\boldsymbol{a}) \cdot s_{k}\right)\right]_{j}
# ```
#
# and can be implemented via automatic differentiation (AD).
#
# This equation is implemented in ExplainabilityMethods as the default method
# for all layer types that don't have a specialized implementation.
# We will refer to it as the "AD fallback".
#
# [^1]: G. Montavon et al., [Layer-Wise Relevance Propagation: An Overview](https://link.springer.com/chapter/10.1007/978-3-030-28954-6_10)
# [^2]: W. Samek et al., [Explaining Deep Neural Networks and Beyond: A Review of Methods and Applications](https://ieeexplore.ieee.org/document/9369420)

# ### AD fallback
# The default LRP fallback for unknown layers uses AD via [Zygote](https://github.com/FluxML/Zygote.jl).
# For `lrp!`, we end up with something that looks very similar to the previous four step computation:
# ```julia
# function lrp!(rule, layer, Rₖ, aₖ, Rₖ₊₁)
#     layerᵨ = modify_layer(rule, layer)
#     c = gradient(aₖ) do a
#             z = layerᵨ(a)
#             s = Zygote.@ignore Rₖ₊₁ ./ modify_denominator(rule, z)
#             z ⋅ s
# 	      end |> only
#     Rₖ .= aₖ .* c
# end
# ```
#
# You can see how `modify_layer` and `modify_denominator` dispatch on the rule and layer type.
# This is how we implemented our own `MyGammaRule`.
# Unknown layers that are registered in the `LRP_CONFIG` use this exact function.

# ### Specialized implementations
# We can also implement specialized versions of `lrp!` based on the type of `layer`,
# e.g. reshaping layers.
#
# Reshaping layers don't affect attributions. We can therefore avoid the computational
# overhead of AD by writing a specialized implementation that simply reshapes back:
# ```julia
# function lrp!(::AbstractLRPRule, ::ReshapingLayer, Rₖ, aₖ, Rₖ₊₁)
#     Rₖ .= reshape(Rₖ₊₁, size(aₖ))
# end
# ```
#
# Since the rule type didn't matter in this case, we didn't specify it.
#
# We can even implement the generic rule as a specialized implementation for `Dense` layers:
# ```julia
# function lrp!(rule::AbstractLRPRule, layer::Dense, Rₖ, aₖ, Rₖ₊₁)
#     ρW, ρb = modify_params(rule, get_params(layer)...)
#     ãₖ₊₁ = modify_denominator(rule, ρW * aₖ + ρb)
#     @tullio Rₖ[j] = aₖ[j] * ρW[k, j] / ãₖ₊₁[k] * Rₖ₊₁[k] # Tullio ≈ fast einsum
# end
# ```
#
# For maximum low-level control beyond `modify_layer`, `modify_params` and `modify_denominator`,
# you can also implement your own `lrp!` function and dispatch
# on individual rule types `MyRule` and layer types `MyLayer`:
# ```julia
# function lrp!(rule::MyRule, layer::MyLayer, Rₖ, aₖ, Rₖ₊₁)
#     Rₖ .= ...
# end
# ```
