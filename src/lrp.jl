"""
    LRP(c::Chain, r::AbstractLRPRule)
    LRP(c::Chain, rs::AbstractVector{<:AbstractLRPRule})

Analyze model by applying Layer-Wise Relevance Propagation.

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
            check_ouput_softmax(model)
            check_model(model; verbose=verbose)
        end
        if length(model.layers) != length(rules)
            throw(ArgumentError("Length of rules doesn't match length of Flux chain."))
        end
        return new{typeof(rules)}(model, rules)
    end
    # Construct LRP analyzer by assigning a single rule to all layers
end

# Additional constructors for convenience:
function LRP(model::Chain, r::AbstractLRPRule; kwargs...)
    model = flatten_model(model)
    rules = repeat([r], length(model.layers))
    return LRP(model, rules; kwargs...)
end
LRPZero(model::Chain; kwargs...) = LRP(model, ZeroRule(); kwargs...)
LRPEpsilon(model::Chain; kwargs...) = LRP(model, EpsilonRule(); kwargs...)
LRPGamma(model::Chain; kwargs...) = LRP(model, GammaRule(); kwargs...)

# The call to the LRP analyzer.
function (analyzer::LRP)(input, ns::AbstractNeuronSelector; layerwise_relevances=false)
    layers = analyzer.model.layers
    acts = Vector{Any}([input])
    # Forward pass through layers, keeping track of activations
    for layer in layers
        append!(acts, [layer(acts[end])])
    end
    rels = deepcopy(acts) # allocate arrays

    # Mask output neuron
    output_neuron = ns(acts[end])
    rels[end] *= 0
    rels[end][output_neuron] = acts[end][output_neuron]

    # Backward pass through layers, applying LRP rules
    for (i, rule) in Iterators.reverse(enumerate(analyzer.rules))
        rels[i] .= rule(layers[i], acts[i], rels[i + 1]) # Rₖ = rule(layer, aₖ, Rₖ₊₁)
    end

    if layerwise_relevances
        return rels, acts
    end

    return rels[1], acts[end] # expl, output
end
