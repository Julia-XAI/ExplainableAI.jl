# # [Getting started](@id docs-getting-started)

#md # !!! note
#md #     This package is part of a wider [Julia XAI ecosystem](https://github.com/Julia-XAI).
#md #     For an introduction to this ecosystem, please refer to the
#md #     [Getting started guide](https://julia-xai.github.io/XAIDocs/).

# For this first example, we already have loaded a pre-trained LeNet5 model
# to look at explanations on the MNIST dataset.
using Flux

using BSON # hide
model = BSON.load("../model.bson", @__MODULE__)[:model] # hide
model

#md # !!! note "Supported models"
#md #
#md #     ExplainableAI.jl can be used on any differentiable classifier.

# ## Preparing the input data
# We use MLDatasets to load a single image from the MNIST dataset:
using MLDatasets
using ImageCore, ImageIO, ImageShow

index = 10
x, y = MNIST(Float32, :test)[10]

convert2image(MNIST, x)

# By convention in Flux.jl, this input needs to be resized to WHCN format
# by adding a color channel and batch dimensions.
input = reshape(x, 28, 28, 1, :);

#md # !!! note "Input format"
#md #
#md #     For any explanation of a model, ExplainableAI.jl assumes the batch dimension
#md #     to come last in the input.
#md #
#md #     For the purpose of heatmapping, the input is assumed to be in WHCN order
#md #     (width, height, channels, batch), which is Flux.jl's convention.

# ## Explanations
# We can now select an analyzer of our choice and call [`analyze`](@ref) to get an [`Attribution`](@ref).
# Note that for gradient-based optimizers, a backend for automatic differentiation must be loaded, by default [Zygote.jl](https://github.com/FluxML/Zygote.jl):
using ExplainableAI
using Zygote

analyzer = InputTimesGradient(model)
attr = analyze(input, analyzer);

# The return value `attr` is of type [`Attribution`](@ref) and bundles the following data:
# * `attr.val`: numerical output of the analyzer, e.g. an attribution or gradient
# * `attr.input`: input the analyzer was applied to
# * `attr.output`: model output for the given analyzer input
# * `attr.output_selection`: index of the output used for the explanation
# * `attr.pooling`: pooling that reduces `attr.val` over its feature dimension,
#     e.g. over the color channels of an image
# * `attr.extras`: optional named tuple that can be used by analyzers
#     to return additional information.
#
# We used `InputTimesGradient`, whose signed attributions are pooled by summation:
attr.pooling

# By default, the explanation is computed for the maximally activated output neuron.
# Since our digit is a 9 and Julia's indexing is 1-based,
# the output neuron at index `10` of our trained model is maximally activated.

# Finally, we obtain the result of the analyzer in form of an array.
attr.val

# ## Heatmapping basics
# Since the array `attr.val` is not very informative at first sight,
# we can visualize `Attribution`s by computing a `heatmap` using either
# [VisionHeatmaps.jl](https://julia-xai.github.io/XAIDocs/VisionHeatmaps/stable/) or
# [TextHeatmaps.jl](https://julia-xai.github.io/XAIDocs/TextHeatmaps/stable/).
using VisionHeatmaps

heatmap(attr)

# If we are only interested in the heatmap, we can combine analysis and heatmapping
# into a single function call:
heatmap(input, analyzer)

# ## Neuron selection
# By passing an additional index to our call to [`analyze`](@ref),
# we can compute an explanation with respect to a specific output neuron.
# Let's see why the output wasn't interpreted as a 4 (output neuron at index 5)
attr = analyze(input, analyzer, 5)
heatmap(attr)

# This heatmap shows us that the "upper loop" of the hand-drawn 9 has negative relevance
# with respect to the output neuron corresponding to digit 4!

#md # !!! note
#md #
#md #     The output neuron can also be specified when calling `heatmap`:
#md #     ```julia
#md #     heatmap(input, analyzer, 5)
#md #     ```

# ## Analyzing batches
# ExplainableAI also supports explanations of input batches:
batchsize = 20
xs, _ = MNIST(Float32, :test)[1:batchsize]
batch = reshape(xs, 28, 28, 1, :) # reshape to WHCN format
attr = analyze(batch, analyzer);

# This will return a single `Attribution` `attr` for the entire batch.
# Calling `heatmap` on `attr` will detect the batch dimension and return a vector of heatmaps.
heatmap(attr)

## Custom heatmaps

# The function `heatmap` automatically picks a preset based on the pooling of the attribution.
#
# Since [`InputTimesGradient`](@ref) computes signed attributions,
# they are summed over color channels and shown in a diverging colormap.
# [`Gradient`](@ref) attributions are pooled by their norm,
# which is non-negative and therefore shown in a sequential colormap:
analyzer = Gradient(model)
heatmap(input, analyzer)
#-
analyzer = InputTimesGradient(model)
heatmap(input, analyzer)

# Using [VisionHeatmaps.jl](https://julia-xai.github.io/XAIDocs/VisionHeatmaps/stable/),
# heatmaps can be heavily customized.
# Check out the [heatmapping documentation](https://julia-xai.github.io/XAIDocs/VisionHeatmaps/stable/) for more information.
