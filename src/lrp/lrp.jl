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
struct LRP{R<:AbstractVector{<:Tuple}} <: AbstractXAIMethod
    model::Chain
    state::R  # each entry is a tuple `(rule, modified_layer)`
end
rules(analyzer::LRP) = map(first, analyzer.state)
modified_layers(analyzer::LRP) = map(last, analyzer.state)

# Construct rule_layer_tuples from model and array of rules
function LRP(
    model::Chain,
    rules::AbstractVector{<:AbstractLRPRule};
    is_flat=false,
    skip_checks=false,
    verbose=true,
)
    !is_flat && (model = flatten_model(model))
    if !skip_checks
        check_output_softmax(model)
        check_model(Val(:LRP), model; verbose=verbose)
    end

    state = map(zip(rules, model.layers)) do (r, l)
        !is_compatible(r, l) && throw(LRPCompatibilityError(r, l))
        return (r, modify_layer(r, l))
    end
    return LRP(model, state)
end

# Construct vector of rules by applying composite
function LRP(model::Chain, composite::Composite; is_flat=false, kwargs...)
    !is_flat && (model = flatten_model(model))
    rules = composite(model)
    return LRP(model, rules; is_flat=true, kwargs...)
end

# Convenience constructor: use ZeroRule everywhere
LRP(model::Chain; kwargs...) = LRP(model, Composite(ZeroRule()); kwargs...)

# The call to the LRP analyzer.
function (analyzer::LRP)(
    input::AbstractArray{T}, ns::AbstractNeuronSelector; layerwise_relevances=false
) where {T}
    layers = analyzer.model.layers
    # Compute layerwise activations on forward pass through model:
    acts = [input, Flux.activations(analyzer.model, input)...]

    # Allocate array for layerwise relevances:
    rels = similar.(acts)

    # Mask output neuron
    output_indices = ns(acts[end])
    rels[end] .= zero(T)
    rels[end][output_indices] = acts[end][output_indices]

    # Backward pass through layers, applying LRP rules
    for (i, (rule, modified_layer)) in Iterators.reverse(enumerate(analyzer.state))
        lrp!(rels[i], rule, modified_layer, acts[i], rels[i + 1]) # inplace update rels[i]
    end

    return Explanation(
        first(rels),
        last(acts),
        output_indices,
        :LRP,
        ifelse(layerwise_relevances, rels, Nothing),
    )
end
