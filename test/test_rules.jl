using ExplainabilityMethods
using ExplainabilityMethods: RULES, modify_params
using Flux
using LinearAlgebra
using ReferenceTests
using Random

# Fixed pseudo-random numbers
T = Float32
pseudorandn(dims...) = randn(MersenneTwister(123), T, dims...)

# Define test layers
ins = 20 # input dimension
outs = 10 # output dimension

layers = Dict(
    "Dense_relu" => Dense(ins, outs, relu; init=pseudorandn),
    "Dense_identity" => Dense(Matrix(I, outs, ins), false, identity),
)

# Define Dense test input
aₖ = pseudorandn(ins)

function approxref(ref::String, act::String)
    # Parse reference string into array and compare
    ref = eval(Meta.parse(ref))
    act = eval(Meta.parse(act))
    return isapprox(ref, act; rtol=0.01)
end

for (rulename, ruletype) in RULES
    rule = ruletype()
    @testset "Rule $rulename" begin
        for (layername, layer) in layers
            @testset "Layer $layername" begin
                Rₖ₊₁ = layer(aₖ)
                Rₖ = rule(layer, aₖ, Rₖ₊₁)

                @test typeof(Rₖ) <: AbstractVector{T}
                @test size(Rₖ) == size(aₖ)

                # println(Rₖ)
                if rulename == "Dense_identity"
                    # First `outs` dimensions should propagate
                    # activations as relevances, rest should be ≈ 0.
                    @test Rₖ[1:outs] ≈ aₖ[1:outs]
                    @test all(Rₖ[outs:end] .< 1e-8)
                end

                @test_reference "references/rules/$(rulename)_$(layername).txt" Rₖ by =
                    approxref
            end
        end
    end
end

# Test individual rules
@testset "modify_params" begin
    W, b = [1.0 -1.0; 2.0 0.0], [-1.0, 1.0]
    ρW, ρb = modify_params(GammaRule(; γ=0.42), W, b)
    @test ρW ≈ [1.42 -1.0; 2.84 0.0]
    @test ρb ≈ [-1.0, 1.42]
end
