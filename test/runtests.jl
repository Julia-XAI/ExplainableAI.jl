using ExplainableAI

using Test
using JuliaFormatter
using Aqua
using JET

@testset "ExplainableAI.jl" begin
    @info "Testing formalities..."
    if VERSION >= v"1.10"
        @testset "Code formatting" begin
            @info "- Testing code formatting with JuliaFormatter..."
            @test JuliaFormatter.format(ExplainableAI; verbose=false, overwrite=false)
        end
        @testset "Aqua.jl" begin
            @info "- Running Aqua.jl tests. These might print warnings from dependencies..."
            Aqua.test_all(ExplainableAI; ambiguities=false)
        end
        @testset "JET tests" begin
            @info "- Testing type stability with JET..."
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
