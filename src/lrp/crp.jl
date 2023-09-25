#=============================#
# CRP struct and constructors #
#=============================#

"""
    CRP(lrp_analyzer, layer, concepts)

Use Concept Relevance Propagation to explain the output of a neural network
with respect to specific neurons in a given layer.

# Arguments
- `lrp_analyzer::LRP`: LRP analyzer
- `layer::Int`: Index of layer in which the concept is located
- `concepts`: Indices of concept neurons at layer `layer_index`.
    Either integer, vector of integers or vector of neuron selectors.

See also [`MaxActivationSelector`](@ref) and [`IndexSelector`](@ref).

# References
[1] R. Achtibat et al., From attribution maps to human-understandable explanations
    through Concept Relevance Propagation
"""
struct CRP{L<:LRP,C<:Vector{<:AbstractNeuronSelector}} <: AbstractXAIMethod
    lrp::L
    layer::Int
    concepts::C

    function CRP(
        lrp::LRP, layer::Int, concepts::C
    ) where {C<:Vector{<:AbstractNeuronSelector}}
        n = length(lrp.model)
        layer ≥ n &&
            throw(ArgumentError("Layer index should be smaller than model length $n"))
        return new{typeof(lrp),typeof(concepts)}(lrp, layer, concepts)
    end
end

# Automatically use IndexSelector when indices are passed
CRP(lrp, layer, concept::Int)       = CRP(lrp, layer, [IndexSelector(concept)])
CRP(lrp, layer, cs::NTuple)         = CRP(lrp, layer, [IndexSelector(c) for c in cs])
CRP(lrp, layer, cs::AbstractVector) = CRP(lrp, layer, [IndexSelector(c) for c in cs])

#======================#
# Call to CRP analyzer #
#======================#

function (crp::CRP)(input::AbstractArray{T,N}, ns::AbstractNeuronSelector) where {T,N}
    rules = crp.lrp.rules
    layers = crp.lrp.model.layers
    modified_layers = crp.lrp.modified_layers

    n_layers = length(layers)
    n_concepts = length(crp.concepts)
    batchsize = size(input, N)

    # Forward pass
    as = get_activations(crp.lrp.model, input) # compute activations aᵏ for all layers k
    Rs = similar.(as)                          # allocate relevances Rᵏ for all layers k
    mask_output_neuron!(Rs[end], as[end], ns)  # compute relevance Rᴺ of output layer N

    # Allocate array for returned relevance, adding concepts to batch dimension
    R_ret = similar(input, size(input)[1:(end - 1)]..., batchsize * n_concepts)
    colons = ntuple(Returns(:), N - 1)

    # Compute regular LRP backward pass until concept layer
    for k in n_layers:-1:(crp.layer + 1)
        lrp!(Rs[k], rules[k], layers[k], modified_layers[k], as[k], Rs[k + 1])
    end

    # Save full relevance at concept layer before masking
    R_copy = deepcopy(Rs[crp.layer + 1])

    # Iterate over concepts
    for (i, concept) in enumerate(crp.concepts)
        mask_concept_neuron!(Rs[crp.layer + 1], R_copy, concept)

        # Continue LRP backward pass
        for k in (crp.layer):-1:1
            lrp!(Rs[k], rules[k], layers[k], modified_layers[k], as[k], Rs[k + 1])
        end

        # Write relevance into R_ret
        start = batchsize * (i - 1) + 1
        stop = batchsize * i
        view(R_ret, colons..., start:stop) .= first(Rs)

        # Reset relevance at concept layer
        if i < n_concepts
            Rs[crp.layer + 1] .= R_copy
        end
    end
    return Explanation(R_ret, last(as), ns(last(as)), :CRP, nothing)
end

# similar to mask_output_neuron! in lrp.jl
function mask_concept_neuron!(R_concept, R, concept_ns::AbstractNeuronSelector)
    idx = concept_ns(R)
    fill!(R_concept, 0)
    R_concept[idx] .= R[idx]
    return R_concept
end
