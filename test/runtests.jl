using ExplainabilityMethods
using Test
using ReferenceTests

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
        println("Running tests on LRP rules...")
        include("test_rules.jl")
    end
    @testset "VGG11" begin
        println("Running tests on VGG11...")
        include("test_vgg11.jl")
    end
end
