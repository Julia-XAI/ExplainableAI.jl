using ExplainableAI
using ADTypes: AutoZygote, AutoEnzyme
using Zygote: Zygote
using Enzyme: Enzyme
using Test

# Regression tests on the number of forward passes through the model (#186).
# Forward passes are counted by a model that increments a global counter on every call.
const FORWARD_PASSES = Ref(0)

function counting_model(x)
    FORWARD_PASSES[] += 1
    return vcat(sum(0.5f0 .* x .^ 2; dims = 1), sum(5 .* x; dims = 1))
end

function count_forward_passes(analyzer, input; kwargs...)
    FORWARD_PASSES[] = 0
    analyze(input, analyzer; kwargs...)
    return FORWARD_PASSES[]
end

# (features = 3, batch = 2)
input = Float32[1.0 -2.0; 2.0 0.5; -3.0 4.0]
input_ref = Float32[0.5 1.0; -1.0 0.0; 2.0 -1.0]
n = 6

# Zygote takes the model output from the forward pass of the differentiation.
@testset "Zygote" begin
    backend = AutoZygote()
    @test count_forward_passes(Gradient(counting_model, backend), input) == 1
    @test count_forward_passes(InputTimesGradient(counting_model, backend), input) == 1

    # One pass on the unaugmented input to select the output, one per sample
    analyzer = SmoothGrad(counting_model, n; backend)
    @test count_forward_passes(analyzer, input) <= n + 1

    # One pass per interpolation point: the input itself is the last point of the path
    analyzer = IntegratedGradients(counting_model, n; backend)
    @test count_forward_passes(analyzer, input) == n
    @test count_forward_passes(analyzer, input; input_ref) == n
end

# Enzyme's split mode selects the output between the forward and the reverse pass.
@testset "Enzyme" begin
    backend = AutoEnzyme()
    @test count_forward_passes(Gradient(counting_model, backend), input) == 1
    @test count_forward_passes(InputTimesGradient(counting_model, backend), input) == 1

    # One pass on the unaugmented input to select the output, one per sample
    analyzer = SmoothGrad(counting_model, n; backend)
    @test count_forward_passes(analyzer, input) <= n + 1

    # One pass per interpolation point: the input itself is the last point of the path
    analyzer = IntegratedGradients(counting_model, n; backend)
    @test count_forward_passes(analyzer, input) == n
    @test count_forward_passes(analyzer, input; input_ref) == n
end
