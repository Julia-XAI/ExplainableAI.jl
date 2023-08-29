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
struct LRP{C<:Chain,R<:ChainTuple,L<:ChainTuple} <: AbstractXAIMethod
    model::C
    rules::R
    modified_layers::L

    # Construct LRP analyzer by assigning a rule to each layer
    function LRP(
        model::Chain, rules::ChainTuple; skip_checks=false, flatten=true, verbose=true
    )
        if flatten
            model = chainflatten(model)
            rules = chainflatten(rules)
        end
        if !skip_checks
            check_output_softmax(model)
            check_lrp_compat(model; verbose=verbose)
        end
        modified_layers = get_modified_layers(rules, model)
        return new{typeof(model),typeof(rules),typeof(modified_layers)}(
            model, rules, modified_layers
        )
    end
end
# Rules can be passed as vector and will be turned to ChainTuple
LRP(model, rules::AbstractVector; kwargs...) = LRP(model, ChainTuple(rules...); kwargs...)

# Convenience constructor without rules: use ZeroRule everywhere
LRP(model::Chain; kwargs...) = LRP(model, Composite(ZeroRule()); kwargs...)

# Construct Chain-/ParallelTuple of rules by applying composite
LRP(model::Chain, c::Composite; kwargs...) = LRP(model, lrp_rules(model, c); kwargs...)

get_activations(model, input) = [input, Flux.activations(model, input)...]

# Call to the LRP analyzer
function (lrp::LRP)(
    input::AbstractArray{T}, ns::AbstractNeuronSelector; layerwise_relevances=false
) where {T}
    acts = get_activations(lrp.model, input)      # compute  aᵏ for all layers k
    rels = similar.(acts)                         # allocate Rᵏ for all layers k
    mask_output_neuron!(rels[end], acts[end], ns) # compute  Rᵏ⁺¹ of output layer

    # Apply LRP rules in backward-pass, inplace-updating relevances `rels[i]`
    for i in length(lrp.model):-1:1
        lrp!(
            rels[i],
            lrp.rules[i],
            lrp.model[i],
            lrp.modified_layers[i],
            acts[i],
            rels[i + 1],
        )
    end
    extras = layerwise_relevances ? (layerwise_relevances=rels,) : nothing

    return Explanation(first(rels), last(acts), ns(last(acts)), :LRP, extras)
end

function lrp!(Rᵏ, rules::ChainTuple, chain::Chain, modified_chain::ChainTuple, aᵏ, Rᵏ⁺¹)
    acts = get_activations(chain, aᵏ)
    rels = similar.(acts)
    last(rels) .= Rᵏ⁺¹

    # Apply LRP rules in backward-pass, inplace-updating relevances `rels[i]`
    for i in length(chain):-1:1
        lrp!(rels[i], rules[i], chain[i], modified_chain[i], acts[i], rels[i + 1])
    end
    return Rᵏ .= first(rels)
end

function lrp!(
    Rᵏ, rules::ParallelTuple, parallel::Parallel, modified_parallel::ParallelTuple, aᵏ, Rᵏ⁺¹
)
    # We re-distribute the relevance Rᵏ⁺¹ to the i-th "branch" of the parallel layer
    # according to the contribution aᵏ⁺¹ᵢ of branch i to the output activation aᵏ⁺¹:
    #   Rᵏ⁺¹ᵢ = Rᵏ⁺¹ .* aᵏ⁺¹ᵢ ./ aᵏ⁺¹ = c .* aᵏ⁺¹ᵢ

    aᵏ⁺¹s = [l(aᵏ) for l in parallel.layers]     # aᵏ⁺¹ᵢ for each branch i
    c = Rᵏ⁺¹ ./ stabilize_denom(sum(aᵏ⁺¹s))
    Rᵏ⁺¹s = [c .* aᵏ⁺¹ᵢ for aᵏ⁺¹ᵢ in aᵏ⁺¹s]      # Rᵏ⁺¹ᵢ for each branch i
    Rᵏs = [similar(aᵏ) for _ in parallel.layers] # pre-allocate output Rᵏᵢ for each branch i

    for (Rᵏᵢ, ruleᵢ, layerᵢ, modified_layerᵢ, Rᵏ⁺¹ᵢ) in
        zip(Rᵏs, rules, parallel.layers, modified_parallel, Rᵏ⁺¹s)
        # In-place update Rᵏᵢ (and therefore Rᵏs)
        lrp!(Rᵏᵢ, ruleᵢ, layerᵢ, modified_layerᵢ, aᵏ, Rᵏ⁺¹ᵢ)
    end
    return Rᵏ .= sum(Rᵏs)
end
