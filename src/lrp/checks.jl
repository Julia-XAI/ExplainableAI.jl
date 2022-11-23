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
    layer_checks_passed = all(layer_checks)
    activ_checks_passed = all(activ_checks)

    layer_names = [_print_name(l) for l in model]
    activ_names = [_print_activation(l) for l in model]
    tab = _summary_table(model, layer_names, layer_checks, activ_names, activ_checks)

    if !layer_checks_passed
        if verbose
            tprint(tab)
            tprint(_layer_check_summary(layer_names, layer_checks))
        end
        error("Unknown layers found in model")
    end
    if !activ_checks_passed
        if verbose
            tprint(tab)
            tprint(_activ_check_summary(activ_names, activ_checks))
        end
        error("Unknown or unsupported activation functions found in model")
    end
    return true # no error
end

function _layer_check_summary(layer_names, layer_checks)
    return Markdown.parse(
        """
        # Layers failed model check
        Found unknown layers `$(join(unique(layer_names[.!layer_checks]), ", "))`
        that are not supported by ExplainableAI's LRP implementation yet.

        If you think the missing layer should be supported by default,
        please [submit an issue](https://github.com/adrhill/ExplainableAI.jl/issues).

        These model checks can be skipped at your own risk
        by setting the LRP-analyzer keyword argument `skip_checks=true`.

        ## Using custom layers
        If you implemented custom layers, register them via
        ```julia
        LRP_CONFIG.supports_layer(::MyLayer) = true         # for structs
        LRP_CONFIG.supports_layer(::typeof(mylayer)) = true # for functions
        ```
        The default fallback for this layer will use Automatic Differentiation
        according to $REF_MONTAVON_OVERVIEW.\\
        """,
    )
end

function _activ_check_summary(activ_names, activ_checks)
    return Markdown.parse(
        """
        # Activations failed model check
        Found layers with unknown  or unsupported activation functions
        `$(join(unique(activ_names[.!activ_checks]), ", "))`.
        LRP assumes that the model is a "deep rectifier network"
        that only contains ReLU-like activation functions.

        If you think the missing activation function should be supported by default,
        please [submit an issue](https://github.com/adrhill/ExplainableAI.jl/issues).

        These model checks can be skipped at your own risk
        by setting the LRP-analyzer keyword argument `skip_checks=true`.

        ## Using custom activation functions
        If you use custom ReLU-like activation functions, register them via
        ```julia
        LRP_CONFIG.supports_activation(::typeof(myfunction)) = true  # for functions
        LRP_CONFIG.supports_activation(::MyActivation) = true        # for structs
        ```
        """,
    )
end

function _summary_table(model::Chain, layer_names, layer_checks, activ_names, activ_checks)
    data = hcat(
        collect(1:length(model)),
        layer_names,
        collect(layer_checks),
        activ_names,
        collect(activ_checks),
    )

    return Table(
        data;
        header=["", "Layer", "Layer supported", "Activation", "Act. supported"],
        box=:ROUNDED,
        header_justify=[:right, :left, :right, :center, :right],
    )
end
