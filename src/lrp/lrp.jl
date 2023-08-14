"""
    LRP(model, rules)
    LRP(model, composite)

Analyze model by applying Layer-Wise Relevance Propagation.
The analyzer can either be created by passing an array of LRP-rules
or by passing a composite, see [`Composite`](@ref) for an example.

# Keyword arguments
- `skip_checks::Bool`: Skip checks whether model is compatible with LRP and contains output softmax. Default is `false`.
- `verbose::Bool`: Select whether the model checks should print a summary on failure. Default is `true`.

# References
[1] G. Montavon et al., Layer-Wise Relevance Propagation: An Overview
[2] W. Samek et al., Explaining Deep Neural Networks and Beyond: A Review of Methods and Applications
"""
struct LRP{R<:ChainTuple} <: AbstractXAIMethod
    model::Chain
    rules::R

    # Construct LRP analyzer by assigning a rule to each layer
    function LRP(model::Chain, rules::ChainTuple; skip_checks=false, verbose=true)
        if !skip_checks
            check_output_softmax(model)
            check_model(Val(:LRP), model; verbose=verbose)
        end
        return new{typeof(rules)}(model, rules)
    end
end
# Rules can be passed as vector and will be turned to ChainTuple
LRP(model, rules::AbstractVector; kwargs...) = LRP(model, ChainTuple(rules...); kwargs...)

# Convenience constructor without rules: use ZeroRule everywhere
LRP(model::Chain; kwargs...) = LRP(model, Composite(ZeroRule()); kwargs...)

# Construct Chain-/ParallelTuple of rules by applying composite
LRP(model::Chain, c::Composite; kwargs...) = LRP(model, lrp_rules(model, c); kwargs...)

# Call to the LRP analyzer
function (lrp::LRP)(
    input::AbstractArray{T}, ns::AbstractNeuronSelector; layerwise_relevances=false
) where {T}
    acts = [input, Flux.activations(lrp.model, input)...] # compute  aₖ for all layers k
    rels = similar.(acts)                                 # allocate Rₖ for all layers k
    mask_output_neuron!(rels[end], acts[end], ns)         # compute  Rₖ₊₁ of output layer

    modified_layers = get_modified_layers(lrp.rules, lrp.model.layers)
    for i in length(lrp.rules):-1:1
        # Backward-pass applying LRP rules, inplace updating rels[i]
        lrp!(rels[i], lrp.rules[i], modified_layers[i], acts[i], rels[i + 1])
    end

    return Explanation(
        first(rels),
        last(acts),
        ns(last(acts)),
        :LRP,
        ifelse(layerwise_relevances, rels, Nothing),
    )
end
