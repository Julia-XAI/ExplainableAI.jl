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

using ExplainableAI
using Zygote

analyzer = InputTimesGradient(model)
attr = analyze(input, analyzer);

attr.pooling

attr.val

using VisionHeatmaps

heatmap(attr)

heatmap(input, analyzer)

attr = analyze(input, analyzer, 5)
heatmap(attr)

batchsize = 20
xs, _ = MNIST(Float32, :test)[1:batchsize]
batch = reshape(xs, 28, 28, 1, :) # reshape to WHCN format
attr = analyze(batch, analyzer);

heatmap(attr)

# Custom heatmaps

analyzer = Gradient(model)
heatmap(input, analyzer)

analyzer = InputTimesGradient(model)
heatmap(input, analyzer)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
