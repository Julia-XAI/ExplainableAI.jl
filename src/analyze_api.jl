abstract type AbstractXAIMethod end
# All analyzers are implemented such that they return an array of explanations:
#   (method::AbstractXAIMethod)(input, ns::AbstractNeuronSelector)::Vector{Explanation}

const BATCHDIM_MISSING = ArgumentError(
    """The input is a 1D vector and therefore missing the required batch dimension.
    Call `analyze` with the keyword argument `add_batch_dim=false`."""
)

"""
    analyze(input, method)
    analyze(input, method, neuron_selection)

Apply the analyzer `method` for the given input, returning an [`Explanation`](@ref).
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

"""
Return type of analyzers when calling [`analyze`](@ref).

## Fields
* `val`: numerical output of the analyzer, e.g. an attribution or gradient
* `output`: model output for the given analyzer input
* `neuron_selection`: neuron index used for the explanation
* `analyzer`: symbol corresponding the used analyzer, e.g. `:LRP` or `:Gradient`
* `extras`: optional named tuple that can be used by analyzers
    to return additional information.
"""
struct Explanation{V,O,I,E<:Union{Nothing,NamedTuple}}
    val::V
    output::O
    neuron_selection::I
    analyzer::Symbol
    extras::E
end
