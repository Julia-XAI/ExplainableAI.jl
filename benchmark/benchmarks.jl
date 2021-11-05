using ExplainabilityMethods
using ExplainabilityMethods: ANALYZERS
using Flux

on_CI = haskey(ENV, "GITHUB_ACTIONS")

include("../test/vgg19.jl")
vgg19 = VGG19(; pretrain=false)
model = flatten_chain(strip_softmax(vgg19.layers))
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
for (name, alg) in algs
    SUITE[name] = BenchmarkGroup(["construct analyzer", "analyze"])
    SUITE[name]["construct analyzer"] = @benchmarkable alg($(model))

    analyzer = alg(model)
    SUITE[name]["analyze"] = @benchmarkable analyze($(img), $(analyzer))
end
