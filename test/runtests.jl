using ExplainabilityMethods
using Test

@testset "ExplainabilityMethods.jl" begin
    @testset "Flux utilities" begin
        include("test_flux_utils.jl")
    end
    @testset "LRP rules" begin
        include("test_rules.jl")
    end
    @testset "Heatmaps" begin
        include("test_heatmaps.jl")
    end
    @testset "VGG-19" begin
        include("test_vgg19.jl")
    end
end
