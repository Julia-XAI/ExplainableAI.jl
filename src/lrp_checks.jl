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
supports_layer(l) = false
supports_layer(::LRPSupportedLayer) = true
"""
    supports_activation(σ)

Check whether LRP can be used on a given activation function.
To extend LRP to your own activation functions, define:
```julia
LRP_CONFIG.supports_activation(::MyActivation) = true
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

"""
    check_model(method::Symbol, model; verbose=true)

Check whether the given method can be used on the model.
Currently, model checks are only implemented for LRP, using the symbol `:LRP`.

# Example
julia> check_model(:LRP, model)
"""
check_model(method::Symbol, model; kwargs...) = check_model(Val(method), model; kwargs...)
function check_model(::Val{:LRP}, c::Chain; verbose=true)
    layer_checks = collect(_check_layer.(Val(:LRP), c.layers))
    activation_checks = collect(_check_activation.(Val(:LRP), c.layers))
    passed_layer_checks = all(layer_checks)
    passed_activation_checks = all(activation_checks)

    passed_layer_checks && passed_activation_checks && return true

    layer_names = [_print_name(l) for l in c]
    activation_names = [_print_activation(l) for l in c]

    verbose && _show_check_summary(
        c, layer_names, layer_checks, activation_names, activation_checks
    )
    if !passed_layer_checks
        verbose && display(
            Markdown.parse(
                """# Layers failed model check
                Found unknown layers `$(join(unique(layer_names[.!layer_checks]), ", "))`
                that are not supported by ExplainabilityMethods' LRP implementation yet.

                If you think the missing layer should be supported by default, please [submit an issue](https://github.com/adrhill/ExplainabilityMethods.jl/issues).

                These model checks can be skipped at your own risk by setting the LRP-analyzer keyword argument `skip_checks=true`.

                ## Using custom layers
                If you implemented custom layers, register them via
                ```julia
                LRP_CONFIG.supports_layer(::MyLayer) = true               # for structs
                LRP_CONFIG.supports_activation(::typeof(mylayer)) = true  # for functions
                ```
                The default fallback for this layer will use Automatic Differentiation according to "Layer-Wise Relevance Propagation: An Overview".
                You can also define a fully LRP-custom rule for your layer by using the interface
                ```julia
                function (rule::AbstractLRPRule)(layer::MyLayer, aₖ, Rₖ₊₁)
                    # ...
                    return Rₖ
                end
                ```
                This pattern can also be used to dispatch on specific rules.
                """,
            ),
        )
        throw(ArgumentError("Unknown layers found in model"))
    end
    if !passed_activation_checks
        verbose && display(
            Markdown.parse(
                """ # Activations failed model check
                Found layers with unknown  or unsupported activation functions
                `$(join(unique(activation_names[.!activation_checks]), ", "))`.
                LRP assumes that the model is a "deep rectifier network" that only contains ReLU-like activation functions.

                If you think the missing activation function should be supported by default, please [submit an issue](https://github.com/adrhill/ExplainabilityMethods.jl/issues).

                These model checks can be skipped at your own risk by setting the LRP-analyzer keyword argument `skip_checks=true`.

                ## Using custom activation functions
                If you use custom ReLU-like activation functions, register them via
                ```julia
                LRP_CONFIG.supports_activation(::typeof(myfunction)) = true  # for functions
                LRP_CONFIG.supports_activation(::MyActivation) = true        # for structs
                ```
                """,
            ),
        )
        throw(ArgumentError("Unknown or unsupported activation functions found in model"))
    end
    return false
end

_print_name(layer) = "$layer"
_print_name(layer::Parallel) = "Parallel(...)"
_print_activation(layer) = hasproperty(layer, :σ) ? "$(layer.σ)" : "—"
_print_activation(layer::Parallel) = "—"

function _show_check_summary(
    c::Chain, layer_names, layer_checks, activation_names, activation_checks
)
    hl_pass = Highlighter((data, i, j) -> j in (3, 5) && data[i, j]; foreground=:green)
    hl_fail = Highlighter((data, i, j) -> j in (3, 5) && !data[i, j]; foreground=:red)
    data = hcat(
        collect(1:length(c)),
        layer_names,
        collect(layer_checks),
        activation_names,
        collect(activation_checks),
    )
    pretty_table(
        data;
        header=["", "Layer", "Layer supported", "Activation", "Act. supported"],
        alignment=[:r, :l, :r, :c, :r],
        highlighters=(hl_pass, hl_fail),
    )
    return nothing
end
