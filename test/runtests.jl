using ExplainableAI

using Test

@testset "ExplainableAI.jl" begin
    @testset verbose = true "Linting" begin
        @info "Running linting tests..."
        include("linting.jl")
    end

    @testset "Constructors" begin
        @info "Testing constructors..."
        include("test_constructors.jl")
    end
    @testset "CNN" begin
        @info "Testing analyzers on CNN..."
        include("test_cnn.jl")
    end
    @testset "Batches" begin
        @info "Testing analyzers on batches..."
        include("test_batches.jl")
    end
    @testset "GPU tests" begin
        include("test_gpu.jl")
    end
    @testset "Benchmark correctness" begin
        @info "Testing whether benchmarks are up-to-date..."
        include("test_benchmarks.jl")
    end
end
