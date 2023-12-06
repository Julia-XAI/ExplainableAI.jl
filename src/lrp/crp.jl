abstract type AbstractCRPConcepts end

#=============================#
# CRP struct and constructors #
#=============================#

"""
    CRP(lrp_analyzer, layer, concepts)

Use Concept Relevance Propagation to explain the output of a neural network
with respect to specific neurons in a given layer.

# Arguments
- `lrp_analyzer::LRP`: LRP analyzer
- `layer::Int`: Index of layer after which the concept is located
- `concepts`: Concept to explain.

See also [`TopNConcepts`](@ref) and [`IndexedConcepts`](@ref).

# References
[1] R. Achtibat et al., From attribution maps to human-understandable explanations
    through Concept Relevance Propagation
"""
struct CRP{L<:LRP,C<:AbstractCRPConcepts} <: AbstractXAIMethod
    lrp::L
    layer::Int
    concepts::C

    function CRP(lrp::LRP, layer::Int, concepts::AbstractCRPConcepts)
        n = length(lrp.model)
        layer ≥ n &&
            throw(ArgumentError("Layer index should be smaller than model length $n"))
        return new{typeof(lrp),typeof(concepts)}(lrp, layer, concepts)
    end
end

#======================#
# Call to CRP analyzer #
#======================#

function (crp::CRP)(input::AbstractArray{T,N}, ns::AbstractNeuronSelector) where {T,N}
    rules = crp.lrp.rules
    layers = crp.lrp.model.layers
    modified_layers = crp.lrp.modified_layers

    n_layers = length(layers)
    n_concepts = number_of_concepts(crp.concepts)
    batchsize = size(input, N)

    # Forward pass
    as = get_activations(crp.lrp.model, input) # compute activations aᵏ for all layers k
    Rs = similar.(as)                          # allocate relevances Rᵏ for all layers k
    mask_output_neuron!(Rs[end], as[end], ns)  # compute relevance Rᴺ of output layer N

    # Allocate array for returned relevance, adding concepts to batch dimension
    R_return = similar(input, size(input)[1:(end - 1)]..., batchsize * n_concepts)
    colons = ntuple(Returns(:), N - 1)

    # Compute regular LRP backward pass until concept layer
    for k in n_layers:-1:(crp.layer + 1)
        lrp!(Rs[k], rules[k], layers[k], modified_layers[k], as[k], Rs[k + 1])
    end

    # Save full relevance at concept layer before masking
    R_concept = Rs[crp.layer + 1]
    R_original = deepcopy(R_concept)

    # Compute neuron indices based on concepts
    concepts_indices = crp.concepts(R_original)

    # Mask concept neurons...
    fill!(R_concept, 0)

    for (i, concept) in enumerate(concepts_indices)
        # ...keeping original relevance at concept neurons
        for idx in concept
            R_concept[idx] .= R_original[idx]
        end

        # Continue LRP backward pass
        for k in (crp.layer):-1:1
            lrp!(Rs[k], rules[k], layers[k], modified_layers[k], as[k], Rs[k + 1])
        end

        # Write relevance into a slice of R_return
        start = batchsize * (i - 1) + 1
        stop = batchsize * i
        view(R_return, colons..., start:stop) .= first(Rs)

        # Reset concept neurons for masking in next iteration
        if i < n_concepts
            for idx in concept
                R_concept[idx] .= 0
            end
        end
    end
    return Explanation(R_return, last(as), ns(last(as)), :CRP, :attribution, nothing)
end

#===================#
# Concept selectors #
#===================#

# Expected interfaces:
# - Calls to concept selector that return a list of lists of CartesianIndices
#   - (c::ConceptSelector)(R::AbstractArray{T,N}) where {T,2}
#   - (c::ConceptSelector)(R::AbstractArray{T,N}) where {T,4}
# - number_of_concepts(c::ConceptSelector)

"""
    IndexedConcepts(indices...)

Select concepts by indices for [`CRP`](@ref).

For outputs of convolutional layers, the index refers to a feature dimension.

See also See also [`IndexedConcepts`](@ref).
"""
struct IndexedConcepts{N} <: AbstractCRPConcepts
    inds::NTuple{N,Int}

    function IndexedConcepts(inds::NTuple{N}) where {N}
        for i in inds
            i > 0 || throw(ArgumentError("All indices have to be greater than 0"))
        end
        return new{N}(inds)
    end
end
IndexedConcepts(args...) = IndexedConcepts(tuple(args...))

number_of_concepts(c::IndexedConcepts) = length(c.inds)

# Pretty printing
Base.show(io::IO, c::IndexedConcepts) = print(io, "IndexedConcepts$(c.inds)")

# Index concepts on 2D arrays, e.g. Dense layers with batch dimension
function (c::IndexedConcepts)(A::AbstractMatrix)
    batchsize = size(A, 2)
    return [[CartesianIndices((i:i, b:b)) for b in 1:batchsize] for i in c.inds]
end

# Index concepts on 4D arrays, e.g. Conv layers with batch dimension
function (c::IndexedConcepts)(A::AbstractArray{T,4}) where {T}
    w, h, _c, batchsize = size(A)
    return [[CartesianIndices((1:w, 1:h, i:i, b:b)) for b in 1:batchsize] for i in c.inds]
end

"""
    TopNConcepts(n)

Select top-n concepts by relevance for [`CRP`](@ref).

For outputs of convolutional layers, the relevance is summed across height and width
channels for each feature.

See also See also [`IndexedConcepts`](@ref).
"""
struct TopNConcepts <: AbstractCRPConcepts
    n::Int

    function TopNConcepts(n)
        n > 0 || throw(ArgumentError("n has to be greater than 0"))
        return new(n)
    end
end

number_of_concepts(c::TopNConcepts) = c.n

# Extract top concepts from 2D arrays, e.g. Dense layers with batch dimension
function (c::TopNConcepts)(A::AbstractMatrix)
    n_features = size(A, 1)
    c.n > n_features && throw(TopNDimensionError(c.n, n_features))

    inds = top_n(A, c.n)
    return [
        [CartesianIndices((i:i, b:b)) for (b, i) in enumerate(r)] for r in eachrow(inds)
    ]
end

# Extract top concepts from 4D array: e.g. Conv layers with batch dimension
function (c::TopNConcepts)(A::AbstractArray{T,4}) where {T}
    w, h, n_features, _batchsize = size(A)
    c.n > n_features && throw(TopNDimensionError(c.n, n_features))

    features = sum(A; dims=(1, 2))[1, 1, :, :] # reduce width and height channels
    inds = top_n(features, c.n)
    return [
        [CartesianIndices((1:w, 1:h, i:i, b:b)) for (b, i) in enumerate(r)] for
        r in eachrow(inds)
    ]
end

function TopNDimensionError(n, nf)
    DimensionMismatch(
        "Attempted to find top $n features, but feature dimensionality is $nf"
    )
end

"""
    top_n(A, n)

For a matrix `A` of size `(rows, batchdims)`, return a matrix of indices of size `(n, batchdims)`
with sorted indices of the top-n entries.

## Example
```julia-repl
julia> A = rand(4, 3)
4×3 Matrix{Float64}:
 0.469809  0.740177  0.100856
 0.96932   0.53207   0.954989
 0.456456  0.837788  0.313662
 0.925512  0.556236  0.0366143

julia> top_n(A, 2)
2×3 Matrix{Int64}:
 2  3  2
 4  1  3
```
"""
top_n(A::AbstractMatrix, n) = mapslices(x -> top_n(x, n), A; dims=1)
top_n(x::AbstractArray, n) = sortperm(x; rev=true)[1:n]
