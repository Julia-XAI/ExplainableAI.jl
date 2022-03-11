using BenchmarkTools
using Flux
using ExplainabilityMethods
import ExplainabilityMethods: _modify_layer

on_CI = haskey(ENV, "GITHUB_ACTIONS")

include("../test/vgg11.jl")
vgg11 = VGG11(; pretrain=false)
model = flatten_model(strip_softmax(vgg11.layers))
img = rand(MersenneTwister(123), Float32, (224, 224, 3, 1))

# Benchmark custom LRP composite
function LRPCustom(model::Chain)
    return LRP(model, [ZBoxRule(), repeat([GammaRule()], length(model.layers) - 1)...])
end

# Use one representative algorithm of each type
algs = Dict(
    "Gradient" => Gradient,
    "InputTimesGradient" => InputTimesGradient,
    "LRPZero" => LRPZero,
    "LRPCustom" => LRPCustom, #modifies weights
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
_modify_layer(r::AbstractLRPRule, w::TestWrapper) = _modify_layer(r, w.layer)
(rule::ZBoxRule)(w::TestWrapper, aₖ, Rₖ₊₁) = rule(w.layer, aₖ, Rₖ₊₁)

# generate input for conv layers
insize = (64, 64, 3, 1)
in_dense = 500
out_dense = 100
aₖ = randn(Float32, insize)

layers = Dict(
    "MaxPool" => (MaxPool((3, 3); pad=0), aₖ),
    "Conv" => (Conv((3, 3), 3 => 2), aₖ),
    "Dense" => (Dense(in_dense, out_dense, relu), randn(Float32, in_dense)),
    "WrappedDense" =>
        (TestWrapper(Dense(in_dense, out_dense, relu)), randn(Float32, in_dense)),
)
rules = Dict(
    "ZeroRule" => ZeroRule(),
    "EpsilonRule" => EpsilonRule(),
    "GammaRule" => GammaRule(),
    "ZBoxRule" => ZBoxRule(),
)

test_rule(rule, layer, aₖ, Rₖ₊₁) = rule(layer, aₖ, Rₖ₊₁) # for use with @benchmarkable macro

SUITE["Layer"] = BenchmarkGroup([k for k in keys(layers)])
for (layername, (layer, aₖ)) in layers
    SUITE["Layer"][layername] = BenchmarkGroup([k for k in keys(rules)])

    Rₖ₊₁ = layer(aₖ)
    for (rulename, rule) in rules
        SUITE["Layer"][layername][rulename] = @benchmarkable test_rule(
            $(rule), $(layer), $(aₖ), $(Rₖ₊₁)
        )
    end
end
