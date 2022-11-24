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
        verbose && _print_model_check(model, layer_checks, activ_checks)
        error("Unknown layers or unsupported activation functions found in model")
    end
    return true # no error
end

#####################################
# Print check summary using Term.jl #
#####################################

const TERM_DW = 6 # difference in width between nested panels

function _print_model_check(model, layer_checks, activ_checks)
    layer_names = [_print_name(l) for l in model]
    activ_names = [_print_activation(l) for l in model]
    tab = _summary_table(model, layer_names, layer_checks, activ_names, activ_checks)

    w = min(100, console_width())
    w_in = w - TERM_DW

    note(title, content) = Panel(content; title=title, style="green", width=w_in)

    warning = RenderableText(_STR_CHECK_FAILED; width=w_in)
    open_issue = RenderableText(_STR_OPEN_ISSUE; width=w_in)
    how_to_skip = RenderableText(_STR_SKIP_CHECK; width=w_in)
    content = [warning, tab, open_issue]
    !all(layer_checks) && push!(content, _panel_custom_layer_help(w_in))
    !all(activ_checks) && push!(content, _panel_custom_activ_help(w_in))
    tprint(
        Panel(
            content...,
            how_to_skip;
            title="FAILED MODEL CHECK",
            style="red",
            box=:DOUBLE,
            width=w,
        ),
    )
    return nothing
end

_STR_CHECK_FAILED = """Found unknown layers or activation functions that are {bold red}not supported{/bold red} by ExplainableAI's LRP implementation yet:"""
_STR_OPEN_ISSUE = """LRP assumes that the model is a deep rectifier network that only contains ReLU-like activation functions.

    {green}If you think the missing layer should be supported by default, {bold}please open an issue{/bold}:{/green}
    https://github.com/adrhill/ExplainableAI.jl/issues
    """
_STR_SKIP_CHECK = """Model checks can be skipped at your own risk by setting the LRP-analyzer keyword argument {blue}skip_checks=true{/blue}."""

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
        header=["", "Layer", "Layer supported", "Act.", "Act. supported"],
        box=:ROUNDED,
        header_justify=[:right, :left, :left, :left, :left],
    )
end

# adapted from Term.jl's parse_md for Markdown.Code content
function code_panel(code, width)
    syntax = highlight_syntax(code)
    return Panel(
        syntax;
        width=width,
        style="white dim on_#262626",
        box=:SQUARE,
        subtitle="julia",
        background="on_#262626",
        subtitle_justify=:right,
    )
end

function _panel_custom_layer_help(width)
    w_in = width - TERM_DW
    return Panel(
        RenderableText("If you implemented custom layers, register them via"; width=w_in),
        code_panel(
            """
            LRP_CONFIG.supports_layer(::MyLayer) = true         # for structs
            LRP_CONFIG.supports_layer(::typeof(mylayer)) = true # for functions""",
            w_in,
        ),
        RenderableText(
            "The default fallback for this layer will use Automatic Differentiation according to $(REF_MONTAVON_OVERVIEW).";
            width=w_in,
        );
        title="Using custom layers",
        style="green",
        width=width,
    )
end
function _panel_custom_activ_help(width)
    w_in = width - TERM_DW
    return Panel(
        RenderableText(
            "If you use custom ReLU-like activation functions, register them via";
            width=w_in,
        ),
        code_panel(
            """
            LRP_CONFIG.supports_activation(::typeof(myfunction)) = true # for functions
            LRP_CONFIG.supports_activation(::MyActivation) = true       # for structs""",
            w_in,
        );
        title="Using custom activation functions",
        style="green",
        width=width,
    )
end
