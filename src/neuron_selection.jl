abstract type AbstractNeuronSelector end
(ns::AbstractNeuronSelector)(output::AbstractArray) = ns(drop_singleton_dims(output))

"""
    MaxActivationNS()

Neuron selector that picks the output neuron with the highest activation.
"""
struct MaxActivationNS <: AbstractNeuronSelector end
(::MaxActivationNS)(output::AbstractVector) = argmax(output)

"""
    IndexNS(index)

Neuron selector that picks the output neuron at the given index.
"""
struct IndexNS{T} <: AbstractNeuronSelector
    index::T
    IndexNS(index::Integer) = new{typeof(index)}(index)
end
(ns::IndexNS)(output::AbstractVector) = ns.index
