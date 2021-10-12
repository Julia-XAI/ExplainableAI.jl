using ExplainabilityMethods
using Flux
using LinearAlgebra
using ReferenceTests
using Random

# Fixed "random" numbers
T = Float64
pseudorandn(dims...) = randn(MersenneTwister(123), T, dims...)

# Define test layers
ins = 20 # input dimension
outs = 10 # output dimension

rules = [ZeroRule(), EpsilonRule(), GammaRule(), ZBoxRule()]
layers = Dict(
    "Dense_relu" => Dense(ins, outs, relu; init=nonrandn),
    "Dense_identity" => Dense(Matrix(I, outs, ins), false, identity),
)

# Define Dense test input
aₖ = pseudorandn(ins)

for rule in rules
    @testset "Rule $(typeof(rule))" begin
        for (layer_name, layer) in layers
            @testset "Layer $(layer_name)" begin
                Rₖ₊₁ = layer(aₖ)
                Rₖ = rule(layer, aₖ, Rₖ₊₁)

                @test typeof(Rₖ) <: AbstractVector{T}
                @test size(Rₖ) == size(aₖ)

                # println(Rₖ)
                if layer_name == "Dense_identity"
                    # First `outs` dimensions should propagate
                    # activations as relevances, rest should be ≈ 0.
                    @test Rₖ[1:outs] ≈ aₖ[1:outs]
                    @test all(Rₖ[outs:end] .< 1e-8)
                end

                @test_reference "references/$(typeof(rule))_$(layer_name).txt" Rₖ
            end
        end
    end
end
