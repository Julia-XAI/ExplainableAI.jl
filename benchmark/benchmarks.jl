using BenchmarkTools
using LoopVectorization
using Flux
using ExplainableAI
import ExplainableAI: lrp!, modify_layer!, get_layer_resetter

on_CI = haskey(ENV, "GITHUB_ACTIONS")

include("../test/vgg11.jl")
vgg11 = VGG11(; pretrain=false)
model = flatten_model(strip_softmax(vgg11.layers))

T = Float32
img = rand(MersenneTwister(123), T, (224, 224, 3, 1))

# Benchmark custom LRP composite
function LRPCustom(model::Chain)
    return LRP(
        model,
        [ZBoxRule(zero(T), oneunit(T)), repeat([GammaRule()], length(model.layers) - 1)...],
    )
end

# Use one representative algorithm of each type
algs = Dict(
    "Gradient" => Gradient,
    "InputTimesGradient" => InputTimesGradient,
    "LRPZero" => LRPZero,
    "LRPCustom" => LRPCustom, #modifies weights
    "SmoothGrad" => model -> SmoothGrad(model, 10),
    "IntegratedGradients" => model -> IntegratedGradients(model, 10),
)

# Define benchmark
contruct_analyzer(alg, model) = alg(model) # for use with @benchmarkable macro

SUITE = BenchmarkGroup()
SUITE["VGG"] = BenchmarkGroup([k for k in keys(algs)])
for (name, alg) in algs
    SUITE["VGG"][name] = BenchmarkGroup(["construct analyzer", "analyze"])
    SUITE["VGG"][name]["construct analyzer"] = @benchmarkable contruct_analyzer(
        $(alg), $(model)
    )

    analyzer = alg(model)
    SUITE["VGG"][name]["analyze"] = @benchmarkable analyze($(img), $(analyzer))
end

# Rules benchmarks – use wrapper to trigger AD fallback
struct TestWrapper{T}
    layer::T
end
(w::TestWrapper)(x) = w.layer(x)
modify_layer!(rule::R, w::TestWrapper) where {R} = modify_layer!(rule, w.layer)
get_layer_resetter(rule::R, w::TestWrapper) where {R} = get_layer_resetter(rule, w.layer)
get_layer_resetter(::ZeroRule, w::TestWrapper) = Returns(nothing)
lrp!(Rₖ, rule::ZBoxRule, w::TestWrapper, aₖ, Rₖ₊₁) = lrp!(Rₖ, rule, w.layer, aₖ, Rₖ₊₁)

# generate input for conv layers
insize = (64, 64, 3, 1)
in_dense = 500
out_dense = 100
aₖ = randn(T, insize)

layers = Dict(
    "MaxPool" => (MaxPool((3, 3); pad=0), aₖ),
    "Conv" => (Conv((3, 3), 3 => 2), aₖ),
    "Dense" => (Dense(in_dense, out_dense, relu), randn(T, in_dense, 1)),
    "WrappedDense" =>
        (TestWrapper(Dense(in_dense, out_dense, relu)), randn(T, in_dense, 1)),
)
rules = Dict(
    "ZeroRule" => ZeroRule(),
    "EpsilonRule" => EpsilonRule(),
    "GammaRule" => GammaRule(),
    "ZBoxRule" => ZBoxRule(zero(T), oneunit(T)),
)

SUITE["Layer"] = BenchmarkGroup([k for k in keys(layers)])
for (layername, (layer, aₖ)) in layers
    SUITE["Layer"][layername] = BenchmarkGroup([k for k in keys(rules)])
    Rₖ = similar(aₖ)
    Rₖ₊₁ = layer(aₖ)
    for (rulename, rule) in rules
        SUITE["Layer"][layername][rulename] = @benchmarkable lrp!(
            $(Rₖ), $(rule), $(layer), $(aₖ), $(Rₖ₊₁)
        )
    end
end
