using ExplainableAI
using ADTypes: AutoZygote, AutoEnzyme
using Zygote: Zygote
using Enzyme: Enzyme, Duplicated
using Test

using Flux: Chain, Dense, relu
using StableRNGs: StableRNG

# Test that analyzers using Enzyme match those using the default Zygote backend.
pseudorand(dims...) = rand(StableRNG(123), Float32, dims...)

input = rand(StableRNG(1), Float32, 5, 2)
input_ref = rand(StableRNG(2), Float32, 5, 2) .- 2

ANALYZERS = Dict(
    "Gradient" => (m, backend) -> Gradient(m, backend),
    "InputTimesGradient" => (m, backend) -> InputTimesGradient(m, backend),
    "SmoothGrad" => (m, backend) -> SmoothGrad(m, 5, 0.1f0, StableRNG(123); backend),
    "IntegratedGradients" => (m, backend) -> IntegratedGradients(m, 5; backend),
)

function test_against_zygote(model, backend_enzyme)
    output = model(input)
    for (name, constructor) in ANALYZERS
        @testset "$name" begin
            kwargs = name == "IntegratedGradients" ? (; input_ref) : (;)
            expl_zygote = analyze(input, constructor(model, AutoZygote()); kwargs...)
            expl_enzyme = analyze(input, constructor(model, backend_enzyme); kwargs...)
            @test expl_enzyme.val ≈ expl_zygote.val
            @test expl_enzyme.output == output
            @test expl_enzyme.output_selection == expl_zygote.output_selection

            # Select the second output
            expl_zygote = analyze(input, constructor(model, AutoZygote()), 2; kwargs...)
            expl_enzyme = analyze(input, constructor(model, backend_enzyme), 2; kwargs...)
            @test expl_enzyme.val ≈ expl_zygote.val
            @test expl_enzyme.output == output
        end
    end
    return nothing
end

model_without_parameters(x) = vcat(sum(0.5f0 .* x .^ 2; dims = 1), sum(5 .* x; dims = 1))

@testset "Function without parameters" begin
    test_against_zygote(model_without_parameters, AutoEnzyme())
end

# Flux models hold their parameters, which requires a shadow of the differentiated function.
@testset "Flux model" begin
    model = Chain(Dense(5 => 8, relu; init = pseudorand), Dense(8 => 3; init = pseudorand))
    test_against_zygote(model, AutoEnzyme(; function_annotation = Duplicated))
end

# Settings of a user-provided reverse mode are kept by the split mode of the extension.
@testset "Reverse mode with runtime activity" begin
    model = Chain(Dense(5 => 8, relu; init = pseudorand), Dense(8 => 3; init = pseudorand))
    mode = Enzyme.set_runtime_activity(Enzyme.Reverse)
    test_against_zygote(model, AutoEnzyme(; mode, function_annotation = Duplicated))
end

# Forward mode isn't covered by the extension and falls back to DifferentiationInterface.jl.
@testset "Forward mode" begin
    test_against_zygote(model_without_parameters, AutoEnzyme(; mode = Enzyme.Forward))
end
