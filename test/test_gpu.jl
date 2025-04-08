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

analyzer_types = (Gradient, SmoothGrad, InputTimesGradient)

@testset "Run analyzer (CPU)" begin
    @testset "$A" for A in analyzer_types
        analyzer = A(model)
        expl = analyze(input, analyzer)
        @test expl isa Explanation
    end
end

@testset "Run analyzer (GPU)" begin
    @testset "$A" for A in analyzer_types
        analyzer_gpu = A(model_gpu)
        expl = analyze(input_gpu, analyzer_gpu)
        @test expl isa Explanation
    end
end
