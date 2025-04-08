using ExplainableAI

using Test
using JuliaFormatter
using Aqua
using JET

@testset "ExplainableAI.jl" begin
    @testset verbose = true "Linting" begin
        @testset "Code formatting" begin
            @info "- running JuliaFormatter code formatting tests..."
            @test JuliaFormatter.format(ExplainableAI; verbose=false, overwrite=false)
        end
        @testset "Aqua.jl" begin
            @info "- running Aqua.jl tests..."
            Aqua.test_all(ExplainableAI; ambiguities=false)
        end
        if VERSION <= v"1.12" # TODO: remove. As of PR #184, JET fails on pre-release builds.
            @testset "JET tests" begin
                @info "- running JET.jl type stability tests..."
                JET.test_package(ExplainableAI; target_defined_modules=true)
            end
        end
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
