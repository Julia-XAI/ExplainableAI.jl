# # [Concept relevance propagation](@id docs-crp)
# In [*From attribution maps to human-understandable explanations through Concept Relevance Propagation*](https://www.nature.com/articles/s42256-023-00711-8) (CRP),
# Achtibat et al. propose the conditioning of LRP relevances on individual features of a model.
#
# This example builds on the basics shown in the [*Getting started*](@ref docs-getting-started) section.
# We start out by loading the same pre-trained LeNet5 model and MNIST input data:
using ExplainableAI
using Flux

using BSON # hide
model = BSON.load("../../model.bson", @__MODULE__)[:model] # hide
model
#-
using MLDatasets
using ImageCore, ImageIO, ImageShow

index = 10
x, y = MNIST(Float32, :test)[10]
input = reshape(x, 28, 28, 1, :)

convert2image(MNIST, x)

# ## Step 1: Create LRP analyzer
# To create a CRP analyzer, first define an LRP analyzer with your desired rules:
composite = EpsilonPlusFlat()
lrp_analyzer = LRP(model, composite)

# ## Step 2: Define concepts
# Then, specify the index of the layer on the outputs of which you want to condition the explanation.
# In this example, we are interested in the outputs of the last convolutional layer, layer 3:
concept_layer = 3    # index of relevant layer in model
model[concept_layer] # show layer

# Then, specify the concepts you are interested in.
# To automatically select the $n$ most relevant concepts, use [`TopNConcepts`](@ref).
#
# Note that for convolutional layers,
# a feature corresponds to an entire output channel of the layer.
concepts = TopNConcepts(5)

# To manually specify features, use [`IndexedConcepts`](@ref).
concepts = IndexedConcepts(1, 2, 10)

# ## Step 3: Use CRP analyzer
# We can now create a [`CRP`](@ref) analyzer
# and use it like any other analyzer from ExplainableAI.jl:
analyzer = CRP(lrp_analyzer, concept_layer, concepts)
heatmap(input, analyzer)

# ## Using CRP on input batches
# Note that `CRP` uses the batch dimension to return explanations.
# When using CRP on batches, the explanations are first sorted by concepts, then inputs,
# e.g. `[c1_i1, c1_i2, c2_i1, c2_i2, c3_i1, c3_i2]` in the following example:
x, y = MNIST(Float32, :test)[10:11]
batch = reshape(x, 28, 28, 1, :)

heatmap(batch, analyzer)
