using ExplainableAI
using ADTypes: AutoZygote, AutoEnzyme, AutoForwardDiff, AutoReverseDiff
using Zygote: Zygote
using Enzyme: Enzyme
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Test

using Flux: Chain, Dense, relu
using StableRNGs: StableRNG

# Zygote and reverse-mode Enzyme are covered by package extensions.
# All other backends use the generic implementation on DifferentiationInterface.jl,
# which has to support reverse-mode as well as forward-mode backends.
# Forward mode is a poor fit for gradients, which is why it is only tested on tiny problems.
BACKENDS = Dict(
    "Zygote" => AutoZygote(),
    "Enzyme" => AutoEnzyme(),
    "Enzyme (forward mode)" => AutoEnzyme(; mode = Enzyme.Forward),
    "ForwardDiff" => AutoForwardDiff(),
    "ReverseDiff" => AutoReverseDiff(),
)

# Analytical correctness tests on a model with known closed-form gradients.
# Two logits make the output selection non-trivial:
#
#     f₁(x) = ½ ∑ᵢ xᵢ²,  f₂(x) = 5 ∑ᵢ xᵢ,    ∂f₁/∂xᵢ = xᵢ,  ∂f₂/∂xᵢ = 5
model_two_logits(x) = vcat(sum(0.5f0 .* x .^ 2; dims = 1), sum(5 .* x; dims = 1))

# (features = 3, batch = 2)
input = Float32[1.0 -2.0; 2.0 0.5; -3.0 4.0]
input_ref = Float32[0.5 1.0; -1.0 0.0; 2.0 -1.0]
output = model_two_logits(input)

# The maximally activated logit is f₁ on the first sample and f₂ on the second.
selection_max = [CartesianIndex(1, 1), CartesianIndex(2, 2)]
@test MaxActivationSelector()(output) == selection_max

grad_f1 = input
grad_f2 = fill(5.0f0, size(input))
grad_max = hcat(grad_f1[:, 1], grad_f2[:, 2])

# Both path gradients are at most linear in α,
# so the trapezoidal rule used by `IntegratedGradients` is exact:
#
#     IG₁ᵢ(x) = ½ (xᵢ² - x'ᵢ²),    IG₂ᵢ(x) = 5 (xᵢ - x'ᵢ)
ig_f1 = 0.5f0 .* (input .^ 2 .- input_ref .^ 2)
ig_f2 = 5 .* (input .- input_ref)
ig_max = hcat(ig_f1[:, 1], ig_f2[:, 2])

@testset "Analytic: $name" for (name, backend) in BACKENDS
    @testset "Gradient" begin
        analyzer = Gradient(model_two_logits, backend)
        expl = analyze(input, analyzer)
        @test expl.val ≈ grad_max
        @test expl.val isa Matrix{Float32}
        @test expl.output == output
        @test expl.output_selection == selection_max
        @test analyze(input, analyzer, 1).val ≈ grad_f1
        @test analyze(input, analyzer, 2).val ≈ grad_f2
    end
    @testset "InputTimesGradient" begin
        analyzer = InputTimesGradient(model_two_logits, backend)
        expl = analyze(input, analyzer)
        @test expl.val ≈ input .* grad_max
        @test expl.val isa Matrix{Float32}
        @test expl.output == output
        @test expl.output_selection == selection_max
        @test analyze(input, analyzer, 1).val ≈ input .* grad_f1
        @test analyze(input, analyzer, 2).val ≈ input .* grad_f2
    end
    @testset "IntegratedGradients" begin
        analyzer = IntegratedGradients(model_two_logits, 5; backend)
        expl = analyze(input, analyzer; input_ref)
        @test expl.val ≈ ig_max
        @test expl.val isa Matrix{Float32}
        @test expl.output == output
        @test expl.output_selection == selection_max
        @test analyze(input, analyzer, 1; input_ref).val ≈ ig_f1
        @test analyze(input, analyzer, 2; input_ref).val ≈ ig_f2
    end
end

# Test that backends match the default Zygote backend on a Flux model.
pseudorand(dims...) = rand(StableRNG(123), Float32, dims...)

@testset "Flux model: $name" for name in ("ForwardDiff", "ReverseDiff")
    backend = BACKENDS[name]
    model = Chain(Dense(3 => 8, relu; init = pseudorand), Dense(8 => 2; init = pseudorand))
    output = model(input)

    analyzers = Dict(
        "Gradient" => b -> Gradient(model, b),
        "InputTimesGradient" => b -> InputTimesGradient(model, b),
        "IntegratedGradients" => b -> IntegratedGradients(model, 5; backend = b),
    )
    @testset "$analyzer_name" for (analyzer_name, constructor) in analyzers
        kwargs = analyzer_name == "IntegratedGradients" ? (; input_ref) : (;)
        expl_zygote = analyze(input, constructor(AutoZygote()); kwargs...)
        expl = analyze(input, constructor(backend); kwargs...)
        @test expl.val ≈ expl_zygote.val
        @test expl.output == output
        @test expl.output_selection == expl_zygote.output_selection

        # Select the second output
        expl_zygote = analyze(input, constructor(AutoZygote()), 2; kwargs...)
        expl = analyze(input, constructor(backend), 2; kwargs...)
        @test expl.val ≈ expl_zygote.val
        @test expl.output == output
    end
end
