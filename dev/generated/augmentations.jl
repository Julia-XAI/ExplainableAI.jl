using ExplainableAI
using Flux

using BSON # hide
model = BSON.load("../model.bson", @__MODULE__)[:model] # hide
model

using MLDatasets
using ImageCore, ImageIO, ImageShow

index = 10
x, y = MNIST(Float32, :test)[10]
input = reshape(x, 28, 28, 1, :)

convert2image(MNIST, x)

analyzer = Gradient(model)
heatmap(input, analyzer)

analyzer = NoiseAugmentation(Gradient(model), 50)
heatmap(input, analyzer)

analyzer = NoiseAugmentation(Gradient(model), 50, 0.1)
heatmap(input, analyzer)

analyzer = SmoothGrad(model, 50)
heatmap(input, analyzer)

using Distributions

analyzer = NoiseAugmentation(Gradient(model), 50, Poisson(0.5))
heatmap(input, analyzer)

analyzer = InterpolationAugmentation(Gradient(model), 50)
heatmap(input, analyzer)

analyzer = IntegratedGradients(model, 50)
heatmap(input, analyzer)

matrix_of_ones = ones(Float32, size(input))

analyzer = InterpolationAugmentation(Gradient(model), 50)
expl = analyzer(input; input_ref=matrix_of_ones)
heatmap(expl)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
