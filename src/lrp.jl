"""
    LRPLayer(layer, rule)

Struct assigning a LRP-rule to a Flux layer.
"""
struct LRPLayer{L, R} where {R<:AbstractLRPRule}
    layer::L
    rule::R
end
# Calls to LRPLayer with one argument are assumed to be calls to the layers:
(l::LRPLayer)(x) = l.layer(x)

# Calls to LRPLayer with two arguments are assumed to be applications of the LRP-rule:
(l::LRPLayer)(aₖ, Rₖ₊₁) = l.rule(l.layer, aₖ, Rₖ₊₁)


"""
    LRP(c::Chain, r::AbstractLRPRule)
    LRP(c::Chain, rs::AbstractVector{<:AbstractLRPRule})
    LRP(layers::AbstractVector{LRPLayer})

Analyzer that applies LRP.
"""
struct LRP{T<:AbstractVector{LRPLayer}}  <: AbstractXAIMethod
    layers::T

    # Construct LRP analyzer that uses a single rule
    function LRP(c::Chain, r::AbstractLRPRule)
        ls = [LRPLayer(l, r) for l in c.layers]
        return new{typeof(ls)}(ls)
    end
    # Construct LRP analyzer by manually assigning a rule to each layer
    function LRP(c::Chain, rs::AbstractVector{<:AbstractLRPRule})
        if length(c.layers) != length(rs)
            throw(DimensionError("Length of rules doesn't match length of Flux chain."))
        end
        ls = [LRPLayer(l, r) for (l, r) in zip(c.layers, rs)]
        return new{typeof(ls)}(ls)
    end
end
# Additional constructors for convenience:
LRPZero(c::Chain) = LRP(c, ZeroRule())
LRPEpsilon(c::Chain) = LRP(c, EpsilonRule())
LRPGamma(c::Chain) = LRP(c, GammaRule())

# The call to the LRP analyzer.
function (analyzer::LRP)(input, ns::AbstractNeuronSelector, ::Val(:LayerwiseRelevances))
    acts = [input,]
    # Forward pass through layers, keeping track of activations
    for l in analyzer.layers
        append!(acts, l(acts[end]))
    end
    rels = acts # allocate arrays

    # Mask output neuron
    output_neuron = ns(activations[end])
    rels[end] .*= 0
    rels[end][output_neuron] .= acts[end][output_neuron]

    # Backward pass through layers, applying LRP rules
    for (i, l) in Iterators.reverse(enumerate(analyzer.layers))
        rels[i] .= l(acts[i], rels[i+1]) # Rₖ = rule(layer, aₖ, Rₖ₊₁)
    end
    return acts, rels
end

function (analyzer::LRP)(input, ns::AbstractNeuronSelector)
    acts, rels = analyzer(input, _output, neuron_selection, :LayerwiseRelevances)
    return acts[end], rels[1] # corresponds to output, expl
end
