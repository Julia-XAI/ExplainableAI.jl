using LayerwiseRelevancePropagation
using Test

@testset "LayerwiseRelevancePropagation.jl" begin
    @testset "VGG-16" begin include("test_vgg16.jl") end
end
