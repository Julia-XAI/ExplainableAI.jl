using ExplainabilityMethods
using Flux
using BSON

model = BSON.load("../model.bson", @__MODULE__)[:model]

using MLDatasets
using ImageCore

index = 10
x, y = MNIST.testdata(Float32, index)

MNIST.convert2image(x)

input = reshape(x, 28, 28, 1, :);

analyzer = LRPZero(model)
expl = analyze(input, analyzer);

heatmap(expl)

heatmap(input, analyzer)

heatmap(input, analyzer, 5)

analyzer = InputTimesGradient(model)
heatmap(input, analyzer)

analyzer = Gradient(model)
heatmap(input, analyzer)

using ColorSchemes
heatmap(expl; cs=ColorSchemes.jet)

heatmap(expl; reduce=:sum, normalize=:extrema, cs=ColorSchemes.inferno)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

