"""
    GradCAM

Analyze model by calculating the gradient of a neuron activation with respect to the input.
This gradient is then used to calculate the Class Activation Map.
"""
struct GradCAM{F,A} <: AbstractXAIMethod
    feature_layers::F
    adaptation_layers::A
end
function (analyzer::GradCAM)(input, ns::AbstractNeuronSelector)
    A = analyzer.feature_layers(input)  # feature map
    feature_map_size = size(A, 1) * size(A, 2)

    # Determine neuron importance αₖᶜ = 1/Z * ∑ᵢ ∑ⱼ ∂yᶜ / ∂Aᵢⱼᵏ
    grad, output, output_indices = gradient_wrt_input(analyzer.adaptation_layers, A, ns)
    αᶜ = sum(grad; dims=(1, 2)) / feature_map_size
    Lᶜ = max.(sum(αᶜ .* A; dims=3), 0)
    return Explanation(Lᶜ, output, output_indices, :GradCAM, :cam, nothing)
end
