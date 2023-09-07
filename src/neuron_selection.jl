abstract type AbstractNeuronSelector end

"""
    MaxActivationSelector()

Neuron selector that picks the output neuron with the highest activation.
"""
struct MaxActivationSelector <: AbstractNeuronSelector end
function (::MaxActivationSelector)(out::AbstractArray{T,N}) where {T,N}
    N < 2 && throw(BATCHDIM_MISSING)
    return vec(argmax(out; dims=1:(N - 1)))
end

"""
    IndexSelector(index)

Neuron selector that picks the output neuron at the given index.
"""
struct IndexSelector{I} <: AbstractNeuronSelector
    index::I
end
function (s::IndexSelector{<:Integer})(out::AbstractArray{T,N}) where {T,N}
    N < 2 && throw(BATCHDIM_MISSING)
    batchsize = size(out, N)
    return [CartesianIndex{N}(s.index, b) for b in 1:batchsize]
end
function (s::IndexSelector{I})(out::AbstractArray{T,N}) where {I,T,N}
    N < 2 && throw(BATCHDIM_MISSING)
    batchsize = size(out, N)
    return [CartesianIndex{N}(s.index..., b) for b in 1:batchsize]
end

"""
    AugmentationSelector(index)

Neuron selector that passes through an augmented neuron selection.
"""
struct AugmentationSelector{I} <: AbstractNeuronSelector
    indices::I
end
(s::AugmentationSelector)(out) = s.indices
