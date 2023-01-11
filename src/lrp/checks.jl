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
supports_activation(σ) = false
supports_activation(::LRPSupportedActivation) = true
end # LRP_CONFIG module

_check_layer(::Val{:LRP}, layer) = LRP_CONFIG.supports_layer(layer)
_check_layer(::Val{:LRP}, c::Chain) = all(_check_layer(Val(:LRP), l) for l in c)

function _check_activation(::Val{:LRP}, layer)
    hasproperty(layer, :σ) && return LRP_CONFIG.supports_activation(layer.σ)
    return true
end
_check_activation(::Val{:LRP}, c::Chain) = all(_check_activation(Val(:LRP), l) for l in c)

# Utils for printing model check summary
_print_name(layer) = "$layer"
_print_name(layer::Parallel) = "Parallel(...)"
_print_activation(layer) = hasproperty(layer, :σ) ? "$(layer.σ)" : "—"
_print_activation(layer::Parallel) = "—"

"""
    check_model(method::Symbol, model; verbose=true)

Check whether the given method can be used on the model.
Currently, model checks are only implemented for LRP, using the symbol `:LRP`.

# Example
julia> check_model(:LRP, model)
"""
check_model(method::Symbol, model; kwargs...) = check_model(Val(method), model; kwargs...)
function check_model(::Val{:LRP}, model::Chain; verbose=true)
    layer_checks = collect(_check_layer.(Val(:LRP), model.layers))
    activ_checks = collect(_check_activation.(Val(:LRP), model.layers))
    if !all(layer_checks) || !all(activ_checks)
        verbose && _display_model_check(model, layer_checks, activ_checks)
        error("Unknown layer or activation function found in model")
    end
    return true # no error
end

function _display_model_check(model, layer_checks, activ_checks)
    layer_names = [_print_name(l) for l in model]
    activ_names = [_print_activation(l) for l in model]

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

const _MD_CHECK_FAILED = md"""# Layers failed model check
    Found unknown layers or activation functions that are not supported
    by ExplainableAI's LRP implementation yet:
    """
const _MD_OPEN_ISSUE = md"""LRP assumes that the model is a deep rectifier network
    that only contains ReLU-like activation functions.

    If you think the missing layer should be supported by default,
    **please [submit an issue](https://github.com/adrhill/ExplainableAI.jl/issues)**.
    """
const _MD_CUSTOM_LAYER = md"""## Using custom layers
    If you implemented custom layers, register them via
    ```julia
    LRP_CONFIG.supports_layer(::MyLayer) = true          # for structs
    LRP_CONFIG.supports_layer(::typeof(mylayer)) = true  # for functions
    ```
    The default fallback for this layer will use Automatic Differentiation
    according to "Layer-Wise Relevance Propagation: An Overview".
    """
const _MD_CUSTOM_ACTIV = md"""## Using custom activation functions
    If you use custom ReLU-like activation functions, register them via
    ```julia
    LRP_CONFIG.supports_activation(::typeof(myfunction)) = true  # for functions
    LRP_CONFIG.supports_activation(::MyActivation) = true        # for structs
    ```
    """
const _MD_SKIP_CHECK = md"""Model checks can be skipped at your own risk by setting
    the `LRP` keyword argument `skip_checks=true`.
    """

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
