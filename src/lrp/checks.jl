module LRP_CONFIG
using ExplainableAI
using ExplainableAI: LRPSupportedLayer, LRPSupportedActivation

# TODO: compatibility with Chains and Parallel layers

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
supports_activation(σ) = false
supports_activation(::LRPSupportedActivation) = true
end # LRP_CONFIG module

lrp_check_layer(layer) = LRP_CONFIG.supports_layer(layer)
lrp_check_layer(c::Chain) = all(lrp_check_layer, c)

# TODO: use activation_fn
function lrp_check_activation(layer)
    hasproperty(layer, :σ) && return LRP_CONFIG.supports_activation(layer.σ)
    return true
end
lrp_check_activation(c::Chain) = all(lrp_check_activation, c)

# Utils for printing model check summary
_print_name(layer) = "$layer"
_print_name(layer::Parallel) = "Parallel(...)"
_print_activation(layer) = hasproperty(layer, :σ) ? "$(layer.σ)" : "—"
_print_activation(layer::Parallel) = "—"

# TODO: document deprecation of check_model
"""
    check_lrp_compat(model; verbose=true)

Check whether LRP can be used on the model.
"""
function check_lrp_compat(model::Chain; verbose=true)
    layer_checks = collect(lrp_check_layer.(model.layers))
    activ_checks = collect(lrp_check_activation.(model.layers))
    if !all(layer_checks) || !all(activ_checks)
        verbose && _display_model_check(model, layer_checks, activ_checks)
        error("Unknown layer or activation function found in model")
    end
    return true # no error
end

function _display_model_check(model, layer_checks, activ_checks)
    layer_names = _print_name.(model)
    activ_names = _print_activation.(model)

    display(_MD_CHECK_FAILED)
    _display_summary_table(layer_names, layer_checks, activ_names, activ_checks)

    message = md""
    push!(message, _MD_OPEN_ISSUE)
    !all(layer_checks) && push!(message, _MD_CUSTOM_LAYER)
    !all(activ_checks) && push!(message, _MD_CUSTOM_ACTIV)
    push!(message, _MD_SKIP_CHECK)
    display(message)
    return nothing
end

# TODO: This could probably be one big string
const _MD_CHECK_FAILED = md"# Layers failed model check
    Found unknown layers or activation functions that are not supported
    by ExplainableAI's LRP implementation yet:
    "
const _MD_OPEN_ISSUE = md"LRP assumes that the model is a deep rectifier network
    that only contains ReLU-like activation functions.

    If you think the missing layer should be supported by default,
    **please [submit an issue](https://github.com/adrhill/ExplainableAI.jl/issues)**.
    "
const _MD_CUSTOM_LAYER = md"""## Using custom layers
    If you implemented custom layers, register them via
    ```julia
    LRP_CONFIG.supports_layer(::MyLayer) = true          # for structs
    LRP_CONFIG.supports_layer(::typeof(mylayer)) = true  # for functions
    ```
    The default fallback for this layer will use Automatic Differentiation
    according to "Layer-Wise Relevance Propagation: An Overview".
    """
const _MD_CUSTOM_ACTIV = md"## Using custom activation functions
    If you use custom ReLU-like activation functions, register them via
    ```julia
    LRP_CONFIG.supports_activation(::typeof(myfunction)) = true  # for functions
    LRP_CONFIG.supports_activation(::MyActivation) = true        # for structs
    ```
    "
const _MD_SKIP_CHECK = md"Model checks can be skipped at your own risk by setting
    the `LRP` keyword argument `skip_checks=true`.
    "

function _display_summary_table(layer_names, layer_checks, activ_names, activ_checks)
    hl_pass = Highlighter((data, i, j) -> j in (3, 5) && data[i, j]; foreground=:green)
    hl_fail = Highlighter((data, i, j) -> j in (3, 5) && !data[i, j]; foreground=:red)
    data = hcat(
        collect(1:length(layer_names)),
        layer_names,
        collect(layer_checks),
        activ_names,
        collect(activ_checks),
    )
    pretty_table(
        data;
        header=["", "Layer", "supported", "Activation", "supported"],
        alignment=[:r, :l, :c, :c, :c],
        highlighters=(hl_pass, hl_fail),
    )
end
