using LoopVectorization
using ExplainableAI
using Flux
using Test
using ReferenceTests
using Aqua
using Random

pseudorand(dims...) = rand(MersenneTwister(123), Float32, dims...)

# Load VGG model:
# Run tests on pseudo-randomly intialized weights to avoid downloading ~550 MB on every CI run.
include("./vgg11.jl")
vgg11 = VGG11(; pretrain=false)

@testset "ExplainableAI.jl" begin
    # Run Aqua.jl quality assurance tests
    @testset "Aqua.jl" begin
        @info "Running Aqua.jl's auto quality assurance tests. These might print warnings from dependencies."
        Aqua.test_all(ExplainableAI; ambiguities=false)
    end

    # Run package tests
    @testset "Utilities" begin
        @info "Testing utilities..."
        include("test_utils.jl")
    end
    @testset "Flux utilities" begin
        @info "Testing chainmap..."
        include("test_chainmap.jl")
    end
    @testset "Neuron selection" begin
        @info "Testing neuron selection..."
        include("test_neuron_selection.jl")
    end
    @testset "Input augmentation" begin
        @info "Testing input augmentation..."
        include("test_input_augmentation.jl")
    end
    @testset "Heatmaps" begin
        @info "Testing heatmaps..."
        include("test_heatmaps.jl")
    end
    @testset "ImageNet preprocessing" begin
        @info "Testing ImageNet preprocessing..."
        include("test_imagenet.jl")
    end
    @testset "Canonize" begin
        @info "Testing model canonization..."
        include("test_canonize.jl")
    end
    @testset "LRP composites" begin
        @info "Testing LRP composites..."
        include("test_composite.jl")
    end
    @testset "LRP model checks" begin
        @info "Testing LRP model checks..."
        include("test_checks.jl")
    end
    @testset "LRP rules" begin
        @info "Testing LRP rules..."
        include("test_rules.jl")
    end
    @testset "Batches" begin
        @info "Testing analyzers on batches..."
        include("test_batches.jl")
    end
    @testset "VGG11" begin
        @info "Testing analyzers on VGG11..."
        include("test_vgg11.jl")
    end
end
