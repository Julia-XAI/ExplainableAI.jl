function gradient_wrt_input(model, input, output_indices)
    return only(gradient((in) -> model(in)[output_indices], input))
end

function gradients_wrt_batch(model, input::AbstractArray{T,N}, output_indices) where {T,N}
    # To avoid computing a sparse jacobian, we compute individual gradients
    # by mapping `gradient_wrt_input` on slices of the input along the batch dimension.
    return mapreduce(
        (gs...) -> cat(gs...; dims=N), zip(eachslice(input; dims=N), output_indices)
    ) do (in, idx)
        gradient_wrt_input(model, batch_dim_view(in), drop_batch_index(idx))
    end
end

"""
    Gradient(model)

Analyze model by calculating the gradient of a neuron activation with respect to the input.
"""
struct Gradient{C<:Chain} <: AbstractXAIMethod
    model::C
    Gradient(model::Chain) = new{typeof(model)}(Flux.testmode!(check_output_softmax(model)))
end
function (analyzer::Gradient)(input, ns::AbstractNeuronSelector)
    output = analyzer.model(input)
    output_indices = ns(output)
    grad = gradients_wrt_batch(analyzer.model, input, output_indices)
    return Explanation(grad, output, output_indices, :Gradient, Nothing)
end

"""
    InputTimesGradient(model)

Analyze model by calculating the gradient of a neuron activation with respect to the input.
This gradient is then multiplied element-wise with the input.
"""
struct InputTimesGradient{C<:Chain} <: AbstractXAIMethod
    model::C
    function InputTimesGradient(model::Chain)
        return new{typeof(model)}(Flux.testmode!(check_output_softmax(model)))
    end
end
function (analyzer::InputTimesGradient)(input, ns::AbstractNeuronSelector)
    output = analyzer.model(input)
    output_indices = ns(output)
    attr = input .* gradients_wrt_batch(analyzer.model, input, output_indices)
    return Explanation(attr, output, output_indices, :InputTimesGradient, Nothing)
end
