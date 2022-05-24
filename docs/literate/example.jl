# # Getting started
# ExplainableAI.jl can be used on any classifier.
# In this first example, we will look at attributions on a LeNet5 model that was pretrained on MNIST.
#
# ### Loading the model
#md # !!! note
#md #
#md #     Outside of these docs, you should be able to load the model using
#md #     ```julia
#md #     using BSON: @load
#md #     @load "model.bson" model
#md #     ```

using ExplainableAI
using Flux
using BSON

model = BSON.load("../model.bson", @__MODULE__)[:model]

#md # !!! warning "Strip softmax"
#md #
#md #     For models with softmax activations on the output, it is necessary to call
#md #     ```julia
#md #     model = strip_softmax(model)
#md #     ```
#md #     before analyzing.

# ### Loading MNIST
# We use MLDatasets to load a single image from the MNIST dataset:
using MLDatasets
using ImageCore
using ImageShow

index = 10
x, y = MNIST.testdata(Float32, index)

MNIST.convert2image(x)

# By convention in Flux.jl, this input needs to be resized to WHCN format by adding a color channel and batch dimensions.
input = reshape(x, 28, 28, 1, :);

#md # !!! warning "Input format"
#md #
#md #     For any attribution of a model, ExplainableAI.jl assumes the batch dimension to be come last in the input.
#md #
#md #     For the purpose of heatmapping, the input is assumed to be in WHCN order
#md #     (width, height, channels, batch), which is Flux.jl's convention.

# ## Calling the analyzer
# We can now select an analyzer of our choice
# and call [`analyze`](@ref) to get an `Explanation`:
analyzer = LRP(model)
expl = analyze(input, analyzer);

# This `Explanation` bundles the following data:
# * `expl.attribution`: the analyzer's attribution
# * `expl.output`: the model output
# * `expl.neuron_selection`: the neuron index of used for the attribution
# * `expl.analyzer`: a symbol corresponding the used analyzer, e.g. `:LRP`

# Finally, we can visualize the `Explanation` through heatmapping:
heatmap(expl)

# Or get the same result by combining both analysis and heatmapping into one step:
heatmap(input, analyzer)

# ## Neuron selection
# By passing an additional index to our call to `analyze`, we can compute the attribution
# with respect to a specific output neuron.
# Let's see why the output wasn't interpreted as a 4 (output neuron at index 5)
heatmap(input, analyzer, 5)

# This heatmap shows us that the "upper loop" of the hand-drawn 9 has negative relevance
# with respect to the output neuron corresponding to digit 4!

#md # !!! note
#md #
#md #     The ouput neuron can also be specified when calling `analyze`:
#md #     ```julia
#md #     expl = analyze(img, analyzer, 5)
#md #     ```

# ## Input batches
# ExplainableAI also supports input batches:
batchsize = 100
xs, _ = MNIST.testdata(Float32, 1:batchsize)
batch = reshape(xs, 28, 28, 1, :) # reshape to WHCN format
expl_batch = analyze(batch, analyzer);

# This will once again return a single `Explanation` `expl_batch` for the entire batch.
# Calling `heatmap` on `expl_batch` will detect the batch dimension and return a vector of heatmaps.
#
# Let's check if the digit at `index = 10` still matches.
hs = heatmap(expl_batch)
hs[index]

# JuliaImages' `mosaic` can be used to return a tiled view of all heatmaps:
mosaic(hs; nrow=10)

# We can also evaluate a batch with respect to a specific output neuron, e.g. for the digit zero at index `1`:
mosaic(heatmap(batch, analyzer, 1); nrow=10)

# ## Automatic heatmap presets
# Currently, the following analyzers are implemented:
#
# ```
# ├── Gradient
# ├── InputTimesGradient
# ├── SmoothGrad
# ├── IntegratedGradients
# └── LRP
#     ├── LRPZero
#     ├── LRPEpsilon
#     └── LRPGamma
# ```
#
# Let's try `InputTimesGradient`
analyzer = InputTimesGradient(model)
heatmap(input, analyzer)

# and `Gradient`
analyzer = Gradient(model)
heatmap(input, analyzer)

# As you can see, the function `heatmap` automatically applies common presets for each method.
#
# Since `InputTimesGradient` and LRP both compute attributions, their presets are similar.
# Gradient methods however are typically shown in grayscale.

# ## Custom heatmap settings
# We can partially or fully override presets by passing keyword arguments to `heatmap`:
using ColorSchemes
heatmap(expl; cs=ColorSchemes.jet)
#
heatmap(expl; reduce=:sum, rangescale=:extrema, cs=ColorSchemes.inferno)

# This also works with batches
mosaic(heatmap(expl_batch; rangescale=:extrema, cs=ColorSchemes.inferno); nrow=10)

# For the full list of keyword arguments, refer to the [`heatmap`](@ref) documentation.
