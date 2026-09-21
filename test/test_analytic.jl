using ExplainableAI
using Zygote
using Test

# Analytical correctness tests on a model with a known closed-form gradient.
#
# The model computes the separable, single-output map
#
#     f(x) = ½ ∑ᵢ xᵢ²     (one logit, batch dimension last)
#
# so that ∂f/∂xᵢ = xᵢ. This gives closed forms for every gradient-based analyzer.
model = x -> sum(0.5f0 .* x .^ 2; dims = 1)

# (features = 3, batch = 2)
input = Float32[1.0 -2.0; 2.0 0.5; -3.0 4.0]

@testset "Gradient" begin
    # ∂f/∂x = x
    @test analyze(input, Gradient(model)).val ≈ input
end

@testset "InputTimesGradient" begin
    # x ⊙ ∂f/∂x = x²
    @test analyze(input, InputTimesGradient(model)).val ≈ input .^ 2
end

@testset "IntegratedGradients (reference input)" begin
    input_ref = Float32[0.5 1.0; -1.0 0.0; 2.0 -1.0]
    input_ref_copy = copy(input_ref)
    analyze(input, IntegratedGradients(model, 5); input_ref = input_ref)
    @test input_ref == input_ref_copy # reference input must not be mutated
end
