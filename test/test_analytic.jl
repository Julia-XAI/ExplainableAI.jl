using ExplainableAI
using ADTypes: AbstractADType, AutoZygote
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

# Zygote takes output and selection from a single forward pass,
# all other backends use a forward pass followed by a vector-Jacobian product.
# Both code paths have to agree. Two logits make the output selection non-trivial:
#
#     f₁(x) = ½ ∑ᵢ xᵢ²,  f₂(x) = 5 ∑ᵢ xᵢ,    ∂f₁/∂xᵢ = xᵢ,  ∂f₂/∂xᵢ = 5
@testset "Gradient code paths" begin
    model_two_logits = x -> vcat(sum(0.5f0 .* x .^ 2; dims = 1), sum(5 .* x; dims = 1))
    output = model_two_logits(input)
    selector = MaxActivationSelector()
    @test selector(output) == [CartesianIndex(1, 1), CartesianIndex(2, 2)]

    gradient_wrt_input = ExplainableAI.gradient_wrt_input
    res_zygote = gradient_wrt_input(model_two_logits, input, selector, AutoZygote())
    res_generic = invoke(
        gradient_wrt_input,
        Tuple{Any, Any, AbstractOutputSelector, AbstractADType},
        model_two_logits, input, selector, AutoZygote(),
    )
    for (grad, out, selection) in (res_zygote, res_generic)
        @test grad ≈ hcat(input[:, 1], fill(5.0f0, 3))
        @test out == output
        @test selection == selector(output)
    end
end

# For this model the path gradient x' + α(x - x') is linear in α, so the n-point
# trapezoidal rule used by `IntegratedGradients` is exact for every n ≥ 2:
#
#     IGᵢ(x) = (xᵢ - x'ᵢ) ∫₀¹ (x'ᵢ + α(xᵢ - x'ᵢ)) dα = ½ (xᵢ² - x'ᵢ²)
#
# It therefore also satisfies the completeness axiom  ∑ᵢ IGᵢ = f(x) - f(x').
@testset "IntegratedGradients (zero reference)" begin
    for n in (2, 5, 10, 50)
        expl = analyze(input, IntegratedGradients(model, n))
        @test expl.val ≈ 0.5f0 .* input .^ 2
        # Completeness: ∑ᵢ IGᵢ = f(x) - f(0)
        @test vec(sum(expl.val; dims = 1)) ≈ vec(model(input))
    end
end

@testset "IntegratedGradients (nonzero reference)" begin
    input_ref = Float32[0.5 1.0; -1.0 0.0; 2.0 -1.0]
    for n in (2, 7, 20)
        input_ref_copy = copy(input_ref)
        expl = analyze(input, IntegratedGradients(model, n); input_ref = input_ref)
        @test input_ref == input_ref_copy # reference input must not be mutated
        @test expl.val ≈ 0.5f0 .* (input .^ 2 .- input_ref .^ 2)
        # Completeness: ∑ᵢ IGᵢ = f(x) - f(x_ref)
        @test vec(sum(expl.val; dims = 1)) ≈ vec(model(input) .- model(input_ref))
    end
end

# A cubic model has a quadratic path gradient, on which the trapezoidal rule is not exact.
# This distinguishes the quadrature rule from alternatives (Riemann sums, plain averages).
#
#     f(x) = ⅓ ∑ᵢ xᵢ³,    ∂f/∂xᵢ = xᵢ²,    IGᵢ(x) = ⅓ (xᵢ³ - x'ᵢ³)
#
# For a quadratic integrand g, the Euler–Maclaurin formula gives the exact error of the
# trapezoidal rule with step h = 1/(n-1):  T(g) = ∫₀¹ g dα + h²/12 (g'(1) - g'(0)).
# With g(α) = (x'ᵢ + αΔᵢ)² and Δᵢ = xᵢ - x'ᵢ, this is  g'(1) - g'(0) = 2Δᵢ², so
#
#     IGᵢ⁽ⁿ⁾(x) = Δᵢ T(g) = ⅓ (xᵢ³ - x'ᵢ³) + h² Δᵢ³ / 6
@testset "IntegratedGradients (trapezoidal rule)" begin
    model_cubic = x -> sum(x .^ 3 ./ 3; dims = 1)
    input_ref = Float32[0.5 1.0; -1.0 0.0; 2.0 -1.0]
    Δ = input .- input_ref
    exact = (input .^ 3 .- input_ref .^ 3) ./ 3
    for n in (2, 3, 10, 50)
        h = 1.0f0 / (n - 1)
        expl = analyze(input, IntegratedGradients(model_cubic, n); input_ref = input_ref)
        @test expl.val ≈ exact .+ h^2 .* Δ .^ 3 ./ 6
    end
    # The completeness gap vanishes as n grows
    gap(n) = maximum(
        abs,
        sum(analyze(input, IntegratedGradients(model_cubic, n); input_ref = input_ref).val; dims = 1) .-
            (model_cubic(input) .- model_cubic(input_ref)),
    )
    @test gap(100) < gap(10) < gap(2)
end
