module LRP_CONFIG
using ExplainabilityMethods
using ExplainabilityMethods: LRPSupportedLayer, LRPSupportedActivation
"""
    supports_layer(layer)

Check whether LRP can be used on a layer or a Chain.
To extend LRP to your own layers, define:
```julia
LRP_CONFIG.supports_layer(::MyLayer) = true
```
"""
supports_layer(l) = (l isa LRPSupportedLayer)
"""
    supports_activation(σ)

Check whether LRP can be used on a given activation function.
To extend LRP to your own activation functions, define:
```julia
LRP_CONFIG.supports_activation(::MyActivation) = true
```
"""
supports_activation(σ) = (σ isa LRPSupportedActivation)
end # LRP_CONFIG module

const LRP_UNKNOWN_ACTIVATION_WARNING = """Found layers with unknown activations.
LRP assumes that the model is a "deep rectifier network" that only contains ReLU-like activation functions.
If you think the missing activation function should be supported by default, please submit an issue:
https://github.com/adrhill/ExplainabilityMethods.jl/issues

These model checks can be skipped at your own risk by setting the LRP-analyzer keyword argument `skip_checks=true`.

# Custom activation functions
If you use custom ReLU-like activation functions, register them via
```julia
LRP_CONFIG.supports_activation(::typeof(myfunction)) = true  # for functions
LRP_CONFIG.supports_activation(::MyActivation) = true        # for structs
```
"""
_check_layer(layer) = LRP_CONFIG.supports_layer(layer)
_check_layer(c::Chain) = all(_check_layer(l) for l in c)

const LRP_UNKNOWN_LAYER_WARNING = """Found unknown layers that are not supported by ExplainabilityMethods' LRP implementation yet.
If you think the missing layer should be supported by default, please submit an issue:
https://github.com/adrhill/ExplainabilityMethods.jl/issues

These model checks can be skipped at your own risk by setting the LRP-analyzer keyword argument `skip_checks=true`.

# Custom layers
If you implemented custom layers, register them via
```julia
LRP_CONFIG.supports_layer(::MyLayer) = true
```
The default fallback for this layer will use Automatic Differentiation according to "Layer-Wise Relevance Propagation: An Overview".
You can define a fully LRP-custom rule for your layer by using the interface
```julia
function (rule::AbstractLRPRule)(layer::MyLayer, aₖ, Rₖ₊₁)
    # ...
    return Rₖ
end
```
"""

function _check_activation(layer)
    hasproperty(layer, :σ) && return LRP_CONFIG.supports_activation(layer.σ)
    return true
end
_check_activation(c::Chain) = all(_check_activation(l) for l in c)

"""
    check_model(model)

Check whether LRP analyzers can be used on the given model.
"""
function check_model(c::Chain)
    layer_checks = _check_layer.(c.layers)
    activation_checks = _check_activation.(c.layers)
    passed_layer_checks = all(layer_checks)
    passed_activation_checks = all(activation_checks)

    passed_layer_checks && passed_activation_checks && return true

    _show_check_summary(c, layer_checks, activation_checks)
    !passed_layer_checks && @warn LRP_UNKNOWN_LAYER_WARNING
    !passed_activation_checks && @warn LRP_UNKNOWN_ACTIVATION_WARNING
    return false
end

_print_name(layer) = "$layer"
_print_name(layer::Parallel) = "Parallel(...)"
function _show_check_summary(c::Chain, layer_checks, activation_checks)
    layernames = [_print_name(l) for l in c]
    data = hcat(
        collect(1:length(c)), layernames, collect(layer_checks), collect(activation_checks)
    )
    pretty_table(
        data; header=["", "Layer", "Layer supported", "Act. supported"], alignment=:l
    )
    return nothing
end
