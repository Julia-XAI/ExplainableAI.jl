using BenchmarkTools
using Flux
using Metalhead
using ExplainabilityMethods

on_CI = haskey(ENV, "GITHUB_ACTIONS")

include("../test/vgg19.jl")
vgg19 = VGG19(; pretrain=false)
model = flatten_model(strip_softmax(vgg19.layers))
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
SUITE = BenchmarkGroup()
SUITE["VGG"] = BenchmarkGroup([k for k in keys(algs)])
for (name, alg) in algs
    SUITE["VGG"][name] = BenchmarkGroup(["construct analyzer", "analyze"])
    SUITE["VGG"][name]["construct analyzer"] = @benchmarkable alg($(model))

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

for (layername, (layer, aₖ)) in layers
    SUITE[layername] = BenchmarkGroup(rulenames)

    for (rulename, ruletype) in rules
        Rₖ₊₁ = layer(aₖ)
        SUITE[layername][rulename] = BenchmarkGroup(["dispatch", "AD fallback"])
        SUITE[layername][rulename]["dispatch"] = @benchmarkable rule($layer, $aₖ, $Rₖ₊₁)
        SUITE[layername][rulename]["AD fallback"] = @benchmarkable rule(
            $TestWrapper(layer), $aₖ, $Rₖ₊₁
        )
    end
end
