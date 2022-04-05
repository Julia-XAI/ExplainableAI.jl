using ExplainableAI
using Test
using ReferenceTests

@testset "ExplainableAI.jl" begin
    @testset "Utilities" begin
        println("Running tests on utilities...")
        include("test_utils.jl")
    end
    @testset "Neuron selection" begin
        println("Running tests on neuron selection...")
        include("test_neuron_selection.jl")
    end
    @testset "Input augmentation" begin
        println("Running tests on input augmentation...")
        include("test_input_augmentation.jl")
    end
    @testset "Heatmaps" begin
        println("Running tests on heatmaps...")
        include("test_heatmaps.jl")
    end
    @testset "Canonize" begin
        println("Running tests on model canonization...")
        include("test_canonize.jl")
    end
    @testset "LRP model checks" begin
        println("Running tests on LRP model checks...")
        include("test_checks.jl")
    end
    @testset "LRP rules" begin
        println("Running tests on LRP rules...")
        include("test_rules.jl")
    end
    @testset "Batches" begin
        println("Running analyzer tests on batches...")
        include("test_batches.jl")
    end
    @testset "VGG11" begin
        println("Running analyzer tests on VGG11...")
        include("test_vgg11.jl")
    end
end
