abstract type AbstractNeuronSelector end

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
struct IndexNS{I<:Integer} <: AbstractNeuronSelector
    index::I
end
(ns::IndexNS)(output::AbstractVector) = ns.index
