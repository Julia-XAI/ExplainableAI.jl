module ExplainableAIZygoteExt

using ExplainableAI: ExplainableAI, AbstractOutputSelector
using ADTypes: AutoZygote
using Zygote: Zygote

# Zygote's pullback separates the forward and reverse passes (#186),
# so the output can be selected after the forward pass to seed the reverse pass.
function ExplainableAI.gradient_wrt_input(
        model, input, selector::AbstractOutputSelector, ::AutoZygote
    )
    output, back = Zygote.pullback(model, input)
    selection = selector(output)
    seed = zero(output)
    seed[selection] .= one(eltype(seed))
    grad = only(back(seed))
    return grad, output, selection
end

end # module
