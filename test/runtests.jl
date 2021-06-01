using ExplainabilityMethods
using Test

@testset "ExplainabilityMethods.jl" begin
    @testset "Reference layers" begin
        include("test_layers.jl")
    end
    # @testset "VGG-19" begin
    #     include("test_vgg19.jl")
    # end
end
