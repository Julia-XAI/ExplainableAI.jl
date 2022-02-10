using ExplainabilityMethods
using Test

@testset "ExplainabilityMethods.jl" begin
    @testset "Utilities" begin
        include("test_utils.jl")
    end
    @testset "Neuron selection" begin
        include("test_neuron_selection.jl")
    end
    @testset "Heatmaps" begin
        include("test_heatmaps.jl")
    end
    @testset "LRP rules" begin
        include("test_rules.jl")
    end
    @testset "VGG-19" begin
        include("test_vgg19.jl")
    end
end
