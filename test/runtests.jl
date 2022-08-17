using LoopVectorization
using ExplainableAI
using Flux
using Test
using ReferenceTests

@testset "ExplainableAI.jl" begin
    @testset "Utilities" begin
        @info "Running tests on utilities..."
        include("test_utils.jl")
    end
    @testset "Neuron selection" begin
        @info "Running tests on neuron selection..."
        include("test_neuron_selection.jl")
    end
    @testset "Input augmentation" begin
        @info "Running tests on input augmentation..."
        include("test_input_augmentation.jl")
    end
    @testset "Heatmaps" begin
        @info "Running tests on heatmaps..."
        include("test_heatmaps.jl")
    end
    @testset "ImageNet preprocessing" begin
        @info "Running tests on ImageNet preprocessing..."
        include("test_imagenet.jl")
    end
    @testset "Canonize" begin
        @info "Running tests on model canonization..."
        include("test_canonize.jl")
    end
    @testset "LRP composites" begin
        @info "Running tests on LRP composites..."
        include("test_composite.jl")
    end
    @testset "LRP model checks" begin
        @info "Running tests on LRP model checks..."
        include("test_checks.jl")
    end
    @testset "LRP rules" begin
        @info "Running tests on LRP rules..."
        include("test_rules.jl")
    end
    @testset "Batches" begin
        @info "Running analyzer tests on batches..."
        include("test_batches.jl")
    end
    @testset "VGG11" begin
        @info "Running analyzer tests on VGG11..."
        include("test_vgg11.jl")
    end
end
