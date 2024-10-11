using BenchmarkTools
using Zygote
using Flux
using ExplainableAI

on_CI = haskey(ENV, "GITHUB_ACTIONS")

T = Float32
input_size = (32, 32, 3, 1)
input = rand(T, input_size)

model = Chain(
    Chain(
        Conv((3, 3), 3 => 8, relu; pad=1),
        Conv((3, 3), 8 => 8, relu; pad=1),
        MaxPool((2, 2)),
        Conv((3, 3), 8 => 16, relu; pad=1),
        Conv((3, 3), 16 => 16, relu; pad=1),
        MaxPool((2, 2)),
    ),
    Chain(
        Flux.flatten,
        Dense(1024 => 512, relu),         # 102_764_544 parameters
        Dropout(0.5),
        Dense(512 => 100, relu),
    ),
)
Flux.testmode!(model, true)

# Use one representative algorithm of each type
METHODS = Dict(
    "Gradient"            => Gradient,
    "InputTimesGradient"  => InputTimesGradient,
    "SmoothGrad"          => model -> SmoothGrad(model, 5),
    "IntegratedGradients" => model -> IntegratedGradients(model, 5),
)

# Define benchmark
construct(method, model) = method(model) # for use with @benchmarkable macro

suite = BenchmarkGroup()
suite["CNN"] = BenchmarkGroup([k for k in keys(METHODS)])
for (name, method) in METHODS
    analyzer = method(model)
    suite["CNN"][name] = BenchmarkGroup(["construct analyzer", "analyze"])
    suite["CNN"][name]["constructor"] = @benchmarkable construct($(method), $(model))
    suite["CNN"][name]["analyze"] = @benchmarkable analyze($(input), $(analyzer))
end
