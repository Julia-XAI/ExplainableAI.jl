using BenchmarkTools
using Flux
using ExplainabilityMethods

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
    SUITE["VGG"][name]["construct analyzer"] = @benchmarkable contruct_analyzer($(alg), $(model))

    analyzer = alg(model)
    SUITE["VGG"][name]["analyze"] = @benchmarkable analyze($(img), $(analyzer))
end

# Rules benchmarks – use wrapper to trigger AD fallback
struct TestWrapper{T}
    layer::T
end
(l::TestWrapper)(x) = l.layer(x)

# generate input for conv layers
insize = (128, 128, 3, 1)
aₖ = randn(Float32, insize)

layers = Dict(
    "MaxPool" => (MaxPool((3, 3); pad=0), aₖ),
    "MeanPool" => (MeanPool((3, 3); pad=0), aₖ),
    "Conv" => (Conv((3, 3), 3 => 6), aₖ),
    "flatten" => (flatten, aₖ),
    "Dense" => (Dense(1000, 200, relu), randn(Float32, 1000)),
)
rules = Dict(
    "ZeroRule" => ZeroRule(),
    "EpsilonRule" => EpsilonRule(),
    "GammaRule" => GammaRule(),
    "ZBoxRule" => ZBoxRule(),
)
rulenames = [k for k in keys(rules)]

test_rule(rule, layer, aₖ, Rₖ₊₁) = rule(layer, aₖ, Rₖ₊₁) # for use with @benchmarkable macro

for (layername, (layer, aₖ)) in layers
    SUITE[layername] = BenchmarkGroup(rulenames)
    Rₖ₊₁ = layer(aₖ)

    for (rulename, rule) in rules
        SUITE[layername][rulename] = BenchmarkGroup(["dispatch", "AD fallback"])
        SUITE[layername][rulename]["dispatch"] = @benchmarkable test_rule($(rule), $(layer), $(aₖ), $(Rₖ₊₁))
        SUITE[layername][rulename]["AD fallback"] = @benchmarkable test_rule($(rule), $(TestWrapper(layer)), $(aₖ), $(Rₖ₊₁))
    end
end
