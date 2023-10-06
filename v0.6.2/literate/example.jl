# # [Getting started](@id docs-getting-started)
# For this first example, we already have loaded a pre-trained LeNet5 model
# to look at explanations on the MNIST dataset.
using ExplainableAI
using Flux

using BSON # hide
model = BSON.load("../model.bson", @__MODULE__)[:model] # hide
model

#md # !!! note "Supported models"
#md #
#md #     ExplainableAI.jl can be used on any differentiable classifier.
#md #
#md #     Only LRP requires models from Flux.jl.

# ## Preparing the model
# For models with softmax activations on the output,
# it is necessary to call [`strip_softmax`](@ref) before analyzing.
model = strip_softmax(model);

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
# We can now select an analyzer of our choice and call [`analyze`](@ref)
# to get an [`Explanation`](@ref):
analyzer = LRP(model)
expl = analyze(input, analyzer);

# The return value `expl` is of type [`Explanation`](@ref) and bundles the following data:
# * `expl.val`: the numerical output of the analyzer, e.g. an attribution or gradient
# * `expl.output`: the model output for the given analyzer input
# * `expl.neuron_selection`: the neuron index used for the explanation
# * `expl.analyzer`: a symbol corresponding the used analyzer, e.g. `:LRP`
# * `expl.extras`: an optional named tuple that can be used by analyzers
#   to return additional information.
#
# We used an LRP analyzer, so `expl.analyzer` is `:LRP`.
expl.analyzer

# By default, the explanation is computed for the maximally activated output neuron.
# Since our digit is a 9 and Julia's indexing is 1-based,
# the output neuron at index `10` of our trained model is maximally activated.
expl.neuron_selection

# Finally, we obtain the result of the analyzer in form of an array.
expl.val

# ## Heatmapping basics
# Since the array `expl.val` is not very informative at first sight,
# we can visualize `Explanation`s by computing a [`heatmap`](@ref):
heatmap(expl)

# If we are only interested in the heatmap, we can combine analysis and heatmapping
# into a single function call:
heatmap(input, analyzer)

# For a more detailed explanation of the `heatmap` function,
# refer to the [heatmapping section](@ref docs-heatmapping).

# ## [List of analyzers](@id docs-analyzers-list)
# Currently, the following analyzers are implemented:
# - [`Gradient`](@ref)
# - [`InputTimesGradient`](@ref)
# - [`SmoothGrad`](@ref)
# - [`IntegratedGradients`](@ref)
# - [`LRP`](@ref)
#   - Rules
#       - [`ZeroRule`](@ref)
#       - [`EpsilonRule`](@ref)
#       - [`GammaRule`](@ref)
#       - [`GeneralizedGammaRule`](@ref)
#       - [`WSquareRule`](@ref)
#       - [`FlatRule`](@ref)
#       - [`ZBoxRule`](@ref)
#       - [`ZPlusRule`](@ref)
#       - [`AlphaBetaRule`](@ref)
#       - [`PassRule`](@ref)
#   - [`Composite`](@ref)
#       - [`EpsilonGammaBox`](@ref)
#       - [`EpsilonPlus`](@ref)
#       - [`EpsilonPlusFlat`](@ref)
#       - [`EpsilonAlpha2Beta1`](@ref)
#       - [`EpsilonAlpha2Beta1Flat`](@ref)

# ## Neuron selection
# By passing an additional index to our call to [`analyze`](@ref),
# we can compute an explanation with respect to a specific output neuron.
# Let's see why the output wasn't interpreted as a 4 (output neuron at index 5)
expl = analyze(input, analyzer, 5)
heatmap(expl)

# This heatmap shows us that the "upper loop" of the hand-drawn 9 has negative relevance
# with respect to the output neuron corresponding to digit 4!

#md # !!! note
#md #
#md #     The output neuron can also be specified when calling [`heatmap`](@ref):
#md #     ```julia
#md #     heatmap(input, analyzer, 5)
#md #     ```

# ## Analyzing batches
# ExplainableAI also supports explanations of input batches:
batchsize = 20
xs, _ = MNIST(Float32, :test)[1:batchsize]
batch = reshape(xs, 28, 28, 1, :) # reshape to WHCN format
expl = analyze(batch, analyzer);

# This will return a single `Explanation` `expl` for the entire batch.
# Calling `heatmap` on `expl` will detect the batch dimension and return a vector of heatmaps.
heatmap(expl)

# For more information on heatmapping batches,
# refer to the [heatmapping documentation](@ref docs-heatmapping-batches).

# ## [GPU support](@id gpu-docs)
# All analyzers support GPU backends,
# building on top of [Flux.jl's GPU support](https://fluxml.ai/Flux.jl/stable/gpu/).
# Using a GPU only requires moving the input array and model weights to the GPU.
#
# For example, using [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl):

# ```julia
# using CUDA, cuDNN
# using Flux
# using ExplainableAI
#
# # move input array and model weights to GPU
# input = input |> gpu # or gpu(input)
# model = model |> gpu # or gpu(model)
#
# # analyzers don't require calling `gpu`
# analyzer = LRP(model)
#
# # explanations are computed on the GPU
# expl = analyze(input, analyzer)
# ```

# Some operations, like saving, require moving explanations back to the CPU.
# This can be done using Flux's `cpu` function:

# ```julia
# val = expl.val |> cpu # or cpu(expl.val)
#
# using BSON
# BSON.@save "explanation.bson" val
# ```
