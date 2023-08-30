# ## Custom layers and activation functions
using Flux
using ExplainableAI

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
model = Chain(Dense(100, 20), MyDoublingLayer())

# Creating an LRP analyzer, e.g. `LRP(model)`, will throw an `ArgumentError`
# and print a summary of the model check in the REPL:
#
# ```julia-repl
# julia> LRP(model)
#   ChainTuple(
#     Dense(100 => 20)  => supported,
#     MyDoublingLayer() => unknown layer type,
#   ),
#
#   LRP model check failed
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡
#
#   Found unknown layer types or activation functions that are not supported by ExplainableAI's LRP implementation yet.
#
#   LRP assumes that the model is a deep rectifier network that only contains ReLU-like activation functions.
#
#   If you think the missing layer should be supported by default, please submit an issue (https://github.com/adrhill/ExplainableAI.jl/issues).
#
#   [...]
#
# ERROR: Unknown layer or activation function found in model
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
model = Chain(Dense(784, 100, myrelu), Dense(100, 10))

# Once again, creating an LRP analyzer for this model will throw an `ArgumentError`
# and display the following model check summary:
#
# ```julia-repl
# julia> LRP(model)
#   ChainTuple(
#     Dense(784 => 100, myrelu) => unsupported or unknown activation function myrelu,
#     Dense(100 => 10)          => supported,
#   ),
#
#   LRP model check failed
#   ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡
#
#   Found unknown layer types or activation functions that are not supported by ExplainableAI's LRP implementation yet.
#
#   LRP assumes that the model is a deep rectifier network that only contains ReLU-like activation functions.
#
#   If you think the missing layer should be supported by default, please submit an issue (https://github.com/adrhill/ExplainableAI.jl/issues).
#
#   [...]
# ```

# Registation works by defining the function [`LRP_CONFIG.supports_activation`](@ref) as `true`:
LRP_CONFIG.supports_activation(::typeof(myrelu)) = true

# now the analyzer can be created without error:
analyzer = LRP(model)
