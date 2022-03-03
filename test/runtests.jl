using ExplainabilityMethods
using Test

@testset "ExplainabilityMethods.jl" begin
    @testset "Utilities" begin
        println("Running tests on utilities...")
        include("test_utils.jl")
    end
    @testset "Neuron selection" begin
        println("Running tests on neuron selection...")
        include("test_neuron_selection.jl")
    end
    @testset "Heatmaps" begin
        println("Running tests on heatmaps...")
        include("test_heatmaps.jl")
    end
    @testset "LRP model checks" begin
        println("Running tests on LRP model checks...")
        include("test_checks.jl")
    end
    @testset "LRP rules" begin
        println("Running tests on rules...")
        include("test_rules.jl")
    end
    @testset "VGG-19" begin
        println("Running tests on VGG16...")
        include("test_vgg19.jl")
    end
end
