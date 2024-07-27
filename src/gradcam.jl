"""
    GradCAM(feature_layers, adaptation_layers)

Calculates the Gradient-weighted Class Activation Map (GradCAM).
GradCAM provides a visual explanation of the regions with significant neuron importance for the model's classification decision.

# Parameters
- `feature_layers`: The layers of a convolutional neural network (CNN) responsible for extracting feature maps.
- `adaptation_layers`: The layers of the CNN used for adaptation and classification.

# Note
Flux is not required for GradCAM. 
GradCAM is compatible with a wide variety of CNN model-families.

# References
- $REF_SELVARAJU_GRADCAM
"""
struct GradCAM{F,A} <: AbstractXAIMethod
    feature_layers::F
    adaptation_layers::A
end
function call_analyzer(input, analyzer::GradCAM, ns::AbstractOutputSelector; kwargs...)
    A = analyzer.feature_layers(input)  # feature map
    feature_map_size = size(A, 1) * size(A, 2)

    # Determine neuron importance αₖᶜ = 1/Z * ∑ᵢ ∑ⱼ ∂yᶜ / ∂Aᵢⱼᵏ
    grad, output, output_indices = gradient_wrt_input(analyzer.adaptation_layers, A, ns)
    αᶜ = sum(grad; dims=(1, 2)) / feature_map_size
    Lᶜ = max.(sum(αᶜ .* A; dims=3), 0)
    return Explanation(Lᶜ, input, output, output_indices, :GradCAM, :cam, nothing)
end
