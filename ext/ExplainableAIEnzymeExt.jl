module ExplainableAIEnzymeExt

using ExplainableAI: ExplainableAI, AbstractOutputSelector
using ADTypes: AutoEnzyme
using Enzyme: Enzyme, Const, Duplicated, ReverseMode, ReverseSplitWithPrimal
using Enzyme.EnzymeCore: Split, WithPrimal

# Enzyme's split mode runs the forward and reverse passes separately (#186),
# so the output can be selected after the forward pass to seed the reverse pass.
# The first `AutoEnzyme` type parameter is the differentiation mode:
# we match reverse mode (or `Nothing`, Enzyme's default), so forward-mode
# Enzyme falls back to the generic method.
function ExplainableAI.gradient_wrt_input(
        model, input, selector::AbstractOutputSelector,
        backend::AutoEnzyme{<:Union{Nothing, ReverseMode}},
    )
    f = annotate_model(model, backend)
    x = Duplicated(input, Enzyme.make_zero(input))
    forward, reverse = Enzyme.autodiff_thunk(
        split_mode(backend), typeof(f), Duplicated, typeof(x)
    )
    tape, output, output_shadow = forward(f, x)
    selection = selector(output)
    fill!(output_shadow, zero(eltype(output_shadow)))
    output_shadow[selection] .= one(eltype(output_shadow))
    reverse(f, x, tape)
    return x.dval, output, selection
end

# The second `AutoEnzyme` type parameter is the `function_annotation`:
# it controls whether the model is differentiated (`Duplicated`, needing a
# parameter shadow) or held constant (`Const`, the default when unset).
annotate_model(model, ::AutoEnzyme{<:Any, <:Union{Nothing, Const}}) = Const(model)
function annotate_model(model, ::AutoEnzyme{<:Any, <:Duplicated})
    return Duplicated(model, Enzyme.make_zero(model))
end

# Turn the mode into a split-mode annotation carrying the primal (needed for the
# output selection). `Nothing` picks Enzyme's default reverse split mode.
split_mode(::AutoEnzyme{Nothing}) = ReverseSplitWithPrimal
function split_mode(backend::AutoEnzyme{<:ReverseMode})
    return WithPrimal(Split(backend.mode))
end

end # module
