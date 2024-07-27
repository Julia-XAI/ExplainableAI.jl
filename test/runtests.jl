using ExplainableAI

using Test
using JuliaFormatter
using Aqua
using JET

@testset "ExplainableAI.jl" begin
    if VERSION >= v"1.10"
        @info "Testing formalities..."
        @testset "Code formatting" begin
            @info "- running JuliaFormatter code formatting tests..."
            @test JuliaFormatter.format(ExplainableAI; verbose=false, overwrite=false)
        end
        @testset "Aqua.jl" begin
            @info "- running Aqua.jl tests. These might print warnings from dependencies..."
            Aqua.test_all(ExplainableAI; ambiguities=false)
        end
        @testset "JET tests" begin
            @info "- running JET.jl type stability tests..."
            JET.test_package(ExplainableAI; target_defined_modules=true)
        end
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
