# # [Heatmapping](@id docs-heatmapping)
# Since numerical explanations are not very informative at first sight,
# we can visualize them by computing a [`heatmap`](@ref).
# This page showcases different options and preset for heatmapping,
# building on the basics shown in the [*Getting started*](@ref docs-getting-started) section.
#
# We start out by loading the same pre-trained LeNet5 model and MNIST input data:
using ExplainableAI
using Flux

using BSON # hide
model = BSON.load("../model.bson", @__MODULE__)[:model] # hide
model
#-
using MLDatasets
using ImageCore, ImageIO, ImageShow

index = 10
x, y = MNIST(Float32, :test)[10]
input = reshape(x, 28, 28, 1, :)

convert2image(MNIST, x)

# ## Automatic heatmap presets
# The function [`heatmap`](@ref) automatically applies common presets for each method.
#
# Since [`InputTimesGradient`](@ref) and [`LRP`](@ref) both compute attributions,
# their presets are similar. Gradient methods however are typically shown in grayscale:
analyzer = Gradient(model)
heatmap(input, analyzer)
#-
analyzer = InputTimesGradient(model)
heatmap(input, analyzer)

# ## Custom heatmap settings
# ### Color schemes
# We can partially or fully override presets by passing keyword arguments to [`heatmap`](@ref).
# For example, we can use a custom color scheme from ColorSchemes.jl using the keyword argument `cs`:
using ColorSchemes

expl = analyze(input, analyzer)
heatmap(expl; cs=ColorSchemes.jet)
#-
heatmap(expl; cs=ColorSchemes.inferno)

# Refer to the [ColorSchemes.jl catalogue](https://juliagraphics.github.io/ColorSchemes.jl/stable/basics/)
# for a gallery of available color schemes.

# ### [Color channel reduction](@id docs-heatmap-reduce)
# Explanations have the same dimensionality as the inputs to the classifier.
# For images with multiple color channels,
# this means that the explanation also has a "color channel" dimension.
#
# The keyword argument `reduce` can be used to reduce this dimension
# to a single scalar value for each pixel.
# The following presets are available:
# - `:sum`: sum up color channels (default setting)
# - `:norm`: compute 2-norm over the color channels
# - `:maxabs`: compute `maximum(abs, x)` over the color channels

heatmap(expl; reduce=:sum)
#-
heatmap(expl; reduce=:norm)
#-
heatmap(expl; reduce=:maxabs)

# Since MNIST only has a single color channel, there is no need for reduction
# and heatmaps look identical.

# ### [Mapping explanations onto the color scheme](@id docs-heatmap-rangescale)
# To map a [color-channel-reduced](@ref docs-heatmap-reduce) explanation onto a color scheme,
# we first need to normalize all values to the range $[0, 1]$.
#
# For this purpose, two presets are available through the `rangescale` keyword argument:
# - `:extrema`: normalize to the minimum and maximum value of the explanation
# - `:centered`: normalize to the maximum absolute value of the explanation.
#   Values of zero will be mapped to the center of the color scheme.
#
# Depending on the color scheme, one of these presets may be more suitable than the other.
# The default color scheme for `InputTimesGradient`, `seismic`, is centered around zero,
# making `:centered` a good choice:
heatmap(expl; rangescale=:centered)
#-
heatmap(expl; rangescale=:extrema)
# However, for the `inferno` color scheme, which is not centered around zero,
# `:extrema` leads to a heatmap with higher contrast.
heatmap(expl; rangescale=:centered, cs=ColorSchemes.inferno)
#-
heatmap(expl; rangescale=:extrema, cs=ColorSchemes.inferno)

# For the full list of `heatmap` keyword arguments, refer to the [`heatmap`](@ref) documentation.

# ## [Heatmapping batches](@id docs-heatmapping-batches)
# Heatmapping also works with input batches.
# Let's demonstrate this by using a batch of 100 images from the MNIST dataset:
xs, ys = MNIST(Float32, :test)[1:100]
batch = reshape(xs, 28, 28, 1, :); # reshape to WHCN format

# The [`heatmap`](@ref) function automatically recognizes
# that the explanation is batched and returns a `Vector` of images:
heatmaps = heatmap(batch, analyzer)

# Image.jl's `mosaic` function can used to display them in a grid:
mosaic(heatmaps; nrow=10)

#md # !!! note "Output type consistency"
#md #
#md #     To obtain a singleton `Vector` containing a single heatmap for non-batched inputs,
#md #     use the `heatmap` keyword argument `unpack_singleton=false`.

# ## Processing heatmaps
# Heatmapping makes use of the
# [Julia-based image processing ecosystem Images.jl](https://github.com/JuliaImages).
#
# If you want to further process heatmaps, you may benefit from reading about some
# [fundamental conventions](https://juliaimages.org/latest/tutorials/quickstart/)
# that the ecosystem utilizes that are different from how images are typically represented
# in OpenCV, MATLAB, ImageJ or Python.

# ## Saving heatmaps
# Since heatmaps are regular Images.jl images, they can be saved as such:
# ```julia
# using FileIO
#
# img = heatmap(input, analyzer)
# save("heatmap.png", img)
# ```
