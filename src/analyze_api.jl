abstract type AbstractXAIMethod end
# All analyzers are implemented such that they return an array of explanations:
#   (method::AbstractXAIMethod)(input, ns::AbstractNeuronSelector)::Vector{Explanation}

const BATCHDIM_MISSING = ArgumentError(
    """The input is a 1D vector and therefore missing the required batch dimension.
    Call analyze with the keyword argument add_batch_dim=false."""
)

"""
    analyze(input, method)
    analyze(input, method, neuron_selection)

Return raw classifier output and explanation.
If `neuron_selection` is specified, the explanation will be calculated for that neuron.
Otherwise, the output neuron with the highest activation is automatically chosen.

## Keyword arguments
- `add_batch_dim`: add batch dimension to the input without allocating. Default is `false`.
"""
function analyze(
    input::AbstractArray{<:Real},
    method::AbstractXAIMethod,
    neuron_selection::Union{Integer,Tuple{<:Integer}};
    kwargs...,
)
    return _analyze(input, method, IndexSelector(neuron_selection); kwargs...)
end

function analyze(input::AbstractArray{<:Real}, method::AbstractXAIMethod; kwargs...)
    return _analyze(input, method, MaxActivationSelector(); kwargs...)
end

function (method::AbstractXAIMethod)(
    input::AbstractArray{<:Real},
    neuron_selection::Union{Integer,Tuple{<:Integer}};
    kwargs...,
)
    return _analyze(input, method, IndexSelector(neuron_selection); kwargs...)
end
function (method::AbstractXAIMethod)(input::AbstractArray{<:Real}; kwargs...)
    return _analyze(input, method, MaxActivationSelector(); kwargs...)
end

# lower-level call to method
function _analyze(
    input::AbstractArray{T,N},
    method::AbstractXAIMethod,
    sel::AbstractNeuronSelector;
    add_batch_dim::Bool=false,
    kwargs...,
) where {T<:Real,N}
    if add_batch_dim
        return method(batch_dim_view(input), sel; kwargs...)
    end
    N < 2 && throw(BATCHDIM_MISSING)
    return method(input, sel; kwargs...)
end

# for convenience, the anaylyzer can be called directly

# Explanations and outputs are returned in a wrapper.
# Metadata such as the analyzer allows dispatching on functions like `heatmap`.
struct Explanation{A,O,I,L}
    attribution::A
    output::O
    neuron_selection::I
    analyzer::Symbol
    layerwise_relevances::L
end
