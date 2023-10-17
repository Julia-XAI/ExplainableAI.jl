using ExplainableAI
using Flux

using BSON # hide
model = BSON.load("../../model.bson", @__MODULE__)[:model] # hide
model

using MLDatasets
using ImageCore, ImageIO, ImageShow

index = 10
x, y = MNIST(Float32, :test)[10]
input = reshape(x, 28, 28, 1, :)

convert2image(MNIST, x)

composite = EpsilonPlusFlat()
lrp_analyzer = LRP(model, composite)

concept_layer = 3    # index of relevant layer in model
model[concept_layer] # show layer

concepts = TopNConcepts(5)

concepts = IndexedConcepts(1, 2, 10)

analyzer = CRP(lrp_analyzer, concept_layer, concepts)
heatmap(input, analyzer)

x, y = MNIST(Float32, :test)[10:11]
batch = reshape(x, 28, 28, 1, :)

heatmap(batch, analyzer)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
