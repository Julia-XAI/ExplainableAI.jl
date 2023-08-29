using ExplainableAI
using Flux
using BSON

model = BSON.load("../model.bson", @__MODULE__)[:model]

using MLDatasets
using ImageCore
using ImageIO
using ImageShow

index = 10
x, _ = MNIST(Float32, :test)[10]

convert2image(MNIST, x)

input = reshape(x, 28, 28, 1, :);

analyzer = LRP(model)
expl = analyze(input, analyzer);

heatmap(expl)

heatmap(input, analyzer)

heatmap(input, analyzer, 5)

batchsize = 100
xs, _ = MNIST(Float32, :test)[1:batchsize]
batch = reshape(xs, 28, 28, 1, :) # reshape to WHCN format
expl_batch = analyze(batch, analyzer);

hs = heatmap(expl_batch)
hs[index]

mosaic(hs; nrow=10)

mosaic(heatmap(batch, analyzer, 1); nrow=10)

analyzer = InputTimesGradient(model)
heatmap(input, analyzer)

analyzer = Gradient(model)
heatmap(input, analyzer)

using ColorSchemes
heatmap(expl; cs=ColorSchemes.jet)

heatmap(expl; reduce=:sum, rangescale=:extrema, cs=ColorSchemes.inferno)

mosaic(heatmap(expl_batch; rangescale=:extrema, cs=ColorSchemes.inferno); nrow=10)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
