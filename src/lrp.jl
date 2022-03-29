"""
    LRP(c::Chain, r::AbstractLRPRule)
    LRP(c::Chain, rs::AbstractVector{<:AbstractLRPRule})

Analyze model by applying Layer-Wise Relevance Propagation.

# Keyword arguments
- `skip_checks::Bool`: Skip checks whether model is compatible with LRP and contains output softmax. Default is `false`.
- `verbose::Bool`: Select whether the model checks should print a summary on failure. Default is `true`.

# References
[1] G. Montavon et al., Layer-Wise Relevance Propagation: An Overview
[2] W. Samek et al., Explaining Deep Neural Networks and Beyond: A Review of Methods and Applications
"""
struct LRP{R<:AbstractVector{<:AbstractLRPRule}} <: AbstractXAIMethod
    model::Chain
    rules::R

    # Construct LRP analyzer by manually assigning a rule to each layer
    function LRP(
        model::Chain,
        rules::AbstractVector{<:AbstractLRPRule};
        skip_checks=false,
        verbose=true,
    )
        model = flatten_model(model)
        if !skip_checks
            check_output_softmax(model)
            check_model(Val(:LRP), model; verbose=verbose)
        end
        if length(model.layers) != length(rules)
            throw(ArgumentError("Length of rules doesn't match length of Flux chain."))
        end
        return new{typeof(rules)}(model, rules)
    end
end

# Construct LRP analyzer by assigning a single rule to all layers
function LRP(model::Chain, r::AbstractLRPRule; kwargs...)
    model = flatten_model(model)
    rules = repeat([r], length(model.layers))
    return LRP(model, rules; kwargs...)
end
# Additional constructors for convenience:
LRP(model::Chain; kwargs...) = LRP(model, ZeroRule(); kwargs...)
LRPZero(model::Chain; kwargs...) = LRP(model, ZeroRule(); kwargs...)
LRPEpsilon(model::Chain; kwargs...) = LRP(model, EpsilonRule(); kwargs...)
LRPGamma(model::Chain; kwargs...) = LRP(model, GammaRule(); kwargs...)

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
    for (i, rule) in Iterators.reverse(enumerate(analyzer.rules))
        lrp!(rule, layers[i], rels[i], acts[i], rels[i + 1]) # inplace update rels[i]
    end

    return Explanation(
        first(rels),
        last(acts),
        output_indices,
        :LRP,
        ifelse(layerwise_relevances, rels, Nothing),
    )
end
