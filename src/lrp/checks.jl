# TODO: document deprecation of check_model / new check_lrp_compat function if exported

module LRP_CONFIG
using ExplainableAI
using ExplainableAI: LRPSupportedLayer, LRPSupportedActivation

"""
    LRP_CONFIG.supports_layer(layer)

Check whether LRP can be used on a layer or a Chain.
To extend LRP to your own layers, define:
```julia
LRP_CONFIG.supports_layer(::MyLayer) = true          # for structs
LRP_CONFIG.supports_layer(::typeof(mylayer)) = true  # for functions
```
"""
supports_layer(l) = false
supports_layer(::LRPSupportedLayer) = true
"""
    LRP_CONFIG.supports_activation(σ)

Check whether LRP can be used on a given activation function.
To extend LRP to your own activation functions, define:
```julia
LRP_CONFIG.supports_activation(::typeof(myactivation)) = true  # for functions
LRP_CONFIG.supports_activation(::MyActivation) = true          # for structs
```
"""
supports_activation(fn) = false
supports_activation(::LRPSupportedActivation) = true
end # LRP_CONFIG module

lrp_check_layer(l) = lrp_check_layer_type(l) && lrp_check_activation(l)

lrp_check_layer_type(l) = LRP_CONFIG.supports_layer(l)

function lrp_check_activation(layer)
    f = activation_fn(layer)
    !isnothing(f) && return LRP_CONFIG.supports_activation(f)
    return true
end

"""
    check_lrp_compat(model; verbose=true)

Check whether LRP can be used on the model.
"""
function check_lrp_compat(model::Chain; verbose=true)
    passed_checks = chainall(lrp_check_layer, model)
    if !passed_checks
        if verbose
            print_lrp_model_check(stdout, model)
            println()
            display(_MD_CHECK_FAILED)
            println()
        end
        error("Unknown layer or activation function found in model")
    end
    return true
end

_MD_CHECK_FAILED = md"""# LRP model check failed

    Found unknown layer types or activation functions that are not supported
    by ExplainableAI's LRP implementation yet.

    LRP assumes that the model is a deep rectifier network
    that only contains ReLU-like activation functions.

    If you think the missing layer should be supported by default,
    **please [submit an issue](https://github.com/adrhill/ExplainableAI.jl/issues)**.

    ## Using custom layers

    If you implemented custom layers, register them via
    ```julia
    LRP_CONFIG.supports_layer(::MyLayer) = true          # for structs
    LRP_CONFIG.supports_layer(::typeof(mylayer)) = true  # for functions
    ```
    The default fallback for this layer will use Automatic Differentiation
    according to *"Layer-Wise Relevance Propagation: An Overview"*.

    ## Using custom activation functions

    If you use custom ReLU-like activation functions, register them via
    ```julia
    LRP_CONFIG.supports_activation(::typeof(myfunction)) = true  # for functions
    LRP_CONFIG.supports_activation(::MyActivation) = true        # for structs
    ```

    ## Skip model checks

    Model checks can be skipped at your own risk by setting
    the `LRP` keyword argument `skip_checks=true`.
    """
