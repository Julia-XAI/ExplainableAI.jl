module ExplainableAIZygoteExt

using ExplainableAI: ExplainableAI, AbstractOutputSelector
using ADTypes: AutoZygote
using Zygote: Zygote

# DifferentiationInterface.jl requires the seed of a pullback ahead of the forward pass,
# whereas the output selection depends on the model output.
# Zygote's pullback separates the two passes:
# the output is selected after the forward pass and seeds the reverse pass (#186).
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
