using ExplainableAI
using Flux

using BSON # hide
model = BSON.load("../model.bson", @__MODULE__)[:model] # hide
model

using MLDatasets
using ImageCore, ImageIO, ImageShow

index = 10
x, y = MNIST(Float32, :test)[10]

convert2image(MNIST, x)

input = reshape(x, 28, 28, 1, :);

analyzer = InputTimesGradient(model)
expl = analyze(input, analyzer);

expl.analyzer

expl.val

heatmap(expl)

heatmap(input, analyzer)

expl = analyze(input, analyzer, 5)
heatmap(expl)

batchsize = 20
xs, _ = MNIST(Float32, :test)[1:batchsize]
batch = reshape(xs, 28, 28, 1, :) # reshape to WHCN format
expl = analyze(batch, analyzer);

heatmap(expl)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
