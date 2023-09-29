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
        # Package extensions break Project.toml formatting tests on Julia 1.6
        # https://github.com/JuliaTesting/Aqua.jl/issues/105
        Aqua.test_all(
            ExplainableAI; ambiguities=false, project_toml_formatting=VERSION >= v"1.7"
        )
    end
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
        include("test_lrp_composite.jl")
    end
    @testset "LRP model checks" begin
        @info "Testing LRP model checks..."
        include("test_lrp_checks.jl")
    end
    @testset "LRP rules" begin
        @info "Testing LRP rules..."
        include("test_lrp_rules.jl")
    end
    @testset "CRP" begin
        @info "Testing CRP..."
        include("test_crp.jl")
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
