using ExplainableAI
using Test

using Flux
using Metal, JLArrays

if Metal.functional()
    @info "Using Metal as GPU device"
    device = mtl # use Apple Metal locally
else
    @info "Using JLArrays as GPU device"
    device = jl # use JLArrays to fake GPU array
end

model = Chain(Dense(10 => 32, relu), Dense(32 => 5))
input = rand(Float32, 10, 8)
@test_nowarn model(input)

model_gpu = device(model)
input_gpu = device(input)
@test_nowarn model_gpu(input_gpu)

analyzer_types = (Gradient, SmoothGrad, InputTimesGradient, IntegratedGradients)

@testset "Run analyzer (CPU)" begin
    @testset "$A" for A in analyzer_types
        analyzer = A(model)
        attr = analyze(input, analyzer)
        @test attr isa Attribution
    end
end

@testset "Run analyzer (GPU)" begin
    @testset "$A" for A in analyzer_types
        analyzer_gpu = A(model_gpu)
        attr = analyze(input_gpu, analyzer_gpu)
        @test attr isa Attribution
    end
end
