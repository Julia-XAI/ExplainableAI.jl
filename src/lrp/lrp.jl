#=============================#
# LRP struct and constructors #
#=============================#

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

#==========================#
# Call to the LRP analyzer #
#==========================#

function (lrp::LRP)(
    input::AbstractArray, ns::AbstractNeuronSelector; layerwise_relevances=false
)
    as = get_activations(lrp.model, input)    # compute activations aᵏ for all layers k
    Rs = similar.(as)                         # allocate relevances Rᵏ for all layers k
    mask_output_neuron!(Rs[end], as[end], ns) # compute relevance Rᴺ of output layer N

    lrp_backward_pass!(Rs, as, lrp.rules, lrp.model, lrp.modified_layers)
    extras = layerwise_relevances ? (layerwise_relevances=Rs,) : nothing
    return Explanation(first(Rs), last(as), ns(last(as)), :LRP, :attribution, extras)
end

get_activations(model, input) = (input, Flux.activations(model, input)...)

function mask_output_neuron!(R_out, a_out, ns::AbstractNeuronSelector)
    fill!(R_out, 0)
    idx = ns(a_out)
    R_out[idx] .= 1
    return R_out
end

function lrp_backward_pass!(Rs, as, rules, layers, modified_layers)
    # Apply LRP rules in backward-pass, inplace-updating relevances `Rs[k]` = Rᵏ
    for k in length(layers):-1:1
        lrp!(Rs[k], rules[k], layers[k], modified_layers[k], as[k], Rs[k + 1])
    end
    return Rs
end

#===========================================#
# Special calls to Flux's "Dataflow layers" #
#===========================================#

function lrp!(Rᵏ, rules::ChainTuple, chain::Chain, modified_chain::ChainTuple, aᵏ, Rᵏ⁺¹)
    as = get_activations(chain, aᵏ)
    Rs = similar.(as)
    last(Rs) .= Rᵏ⁺¹

    lrp_backward_pass!(Rs, as, rules, chain, modified_chain)
    return Rᵏ .= first(Rs)
end

function lrp!(
    Rᵏ, rules::ParallelTuple, parallel::Parallel, modified_parallel::ParallelTuple, aᵏ, Rᵏ⁺¹
)
    # We re-distribute the relevance Rᵏ⁺¹ to the i-th "branch" of the parallel layer
    # according to the contribution aᵏ⁺¹ᵢ of branch i to the output activation aᵏ⁺¹:
    #   Rᵏ⁺¹ᵢ = Rᵏ⁺¹ .* aᵏ⁺¹ᵢ ./ aᵏ⁺¹ = c .* aᵏ⁺¹ᵢ

    aᵏ⁺¹_parallel = [layer(aᵏ) for layer in parallel.layers] # aᵏ⁺¹ᵢ for each branch i
    c = Rᵏ⁺¹ ./ stabilize_denom(sum(aᵏ⁺¹_parallel))
    Rᵏ⁺¹_parallel = [c .* a for a in aᵏ⁺¹_parallel]          # Rᵏ⁺¹ᵢ for each branch i
    Rᵏ_parallel = [similar(aᵏ) for _ in parallel.layers]     # pre-allocate output Rᵏᵢ for each branch

    for (Rᵏᵢ, rule, layer, modified_layer, Rᵏ⁺¹ᵢ) in
        zip(Rᵏ_parallel, rules, parallel.layers, modified_parallel, Rᵏ⁺¹_parallel)
        # In-place update Rᵏᵢ and therefore Rᵏ_parallel
        lrp!(Rᵏᵢ, rule, layer, modified_layer, aᵏ, Rᵏ⁺¹ᵢ)
    end
    return Rᵏ .= sum(Rᵏ_parallel)
end
