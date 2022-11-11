using BenchmarkTools
using LoopVectorization
using Flux
using ExplainableAI
using ExplainableAI: lrp!, modify_layer

on_CI = haskey(ENV, "GITHUB_ACTIONS")

include("../test/vgg11.jl")
vgg11 = VGG11(; pretrain=false)
model = flatten_model(strip_softmax(vgg11.layers))

T = Float32
img = rand(MersenneTwister(123), T, (224, 224, 3, 1))

# Use one representative algorithm of each type
algs = Dict(
    "Gradient"            => Gradient,
    "InputTimesGradient"  => InputTimesGradient,
    "LRP"                 => LRP,
    "LREpsilonPlusFlat"   => model -> LRP(model, EpsilonPlusFlat()),
    "SmoothGrad"          => model -> SmoothGrad(model, 5),
    "IntegratedGradients" => model -> IntegratedGradients(model, 5),
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

# generate input for conv layers
insize = (64, 64, 3, 1)
in_dense = 500
out_dense = 100
aₖ = randn(T, insize)

layers = Dict(
    "Conv"  => (Conv((3, 3), 3 => 2), aₖ),
    "Dense" => (Dense(in_dense, out_dense, relu), randn(T, in_dense, 1)),
)
rules = Dict(
    "ZeroRule"      => ZeroRule(),
    "EpsilonRule"   => EpsilonRule(),
    "GammaRule"     => GammaRule(),
    "WSquareRule"   => WSquareRule(),
    "FlatRule"      => FlatRule(),
    "AlphaBetaRule" => AlphaBetaRule(),
    "ZPlusRule"     => ZPlusRule(),
    "ZBoxRule"      => ZBoxRule(zero(T), oneunit(T)),
)

SUITE["Layer"] = BenchmarkGroup([k for k in keys(layers)])
for (layername, (layer, aₖ)) in layers
    SUITE["Layer"][layername] = BenchmarkGroup([k for k in keys(rules)])
    Rₖ = similar(aₖ)
    Rₖ₊₁ = layer(aₖ)
    for (rulename, rule) in rules
        SUITE["Layer"][layername][rulename] = BenchmarkGroup(["modify layer", "apply rule"])
        SUITE["Layer"][layername][rulename]["modify layer"] = @benchmarkable modify_layer(
            $(rule), $(layer)
        )
        modified_layer = modify_layer(rule, layer)
        SUITE["Layer"][layername][rulename]["apply rule"] = @benchmarkable lrp!(
            $(Rₖ), $(rule), $(modified_layer), $(aₖ), $(Rₖ₊₁)
        )
    end
end
