using ExplainabilityMethods
using Flux
using LinearAlgebra
using ReferenceTests
using Random

# Fixed "random" numbers
T= Float64
nonrandn(dims...) = randn(MersenneTwister(123), T, dims...)

# Define test layers
ins = 20 # input dimension
outs = 10 # output dimension

rules = Dict(
    "LRP-0" => LRP_0,
    "LRP-ϵ" => LRP_ϵ,
    "LRP-γ" => LRP_γ,
    "LRP-zᴮ" => LRP_zᴮ,
)
layers = Dict(
    "Dense_ReLU" => Dense(ins, outs, relu; init=nonrandn),
    "Dense_identity" => Dense(Matrix(I, outs, ins), false, identity)
)

# Define Dense test input
aₖ = nonrandn(ins)

for (rule_name, rule) in rules
    @testset "Rule $(rule_name)" begin
        for (layer_name, layer) in layers
            @testset "Layer $(layer_name)" begin
                analyzer = rule(layer; T=T)

                # Apply layer
                Rₖ₊₁ = layer(aₖ)
                Rₖ = analyzer(aₖ, Rₖ₊₁)

                @test typeof(Rₖ) <: AbstractVector{T}
                @test size(Rₖ) == size(aₖ)

                # println(Rₖ)
                if layer_name == "Dense_identity"
                    # First `outs` dimensions should propagate
                    # activations as relevances, rest should be ≈ 0.
                    @test all(Rₖ[1:outs] .≈ aₖ[1:outs])
                    @test all(Rₖ[outs:end] .< 1e-8)
                end

                @test_reference "references/R_$(layer_name)_$(rule_name).txt" Rₖ
            end
        end
    end
end
