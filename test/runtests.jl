using LoopVectorization
using ExplainableAI
using Flux
using Test
using ReferenceTests
using Aqua
using Random

pseudorand(dims...) = rand(MersenneTwister(123), Float32, dims...)

@testset "ExplainableAI.jl" begin
    @testset "Aqua.jl" begin
        @info "Running Aqua.jl's auto quality assurance tests. These might print warnings from dependencies."
        Aqua.test_all(ExplainableAI; ambiguities=false)
    end
    @testset "Input augmentation" begin
        @info "Testing input augmentation..."
        include("test_input_augmentation.jl")
    end
    @testset "CNN" begin
        @info "Testing analyzers on CNN..."
        include("test_cnn.jl")
    end
    @testset "Batches" begin
        @info "Testing analyzers on batches..."
        include("test_batches.jl")
    end
end
