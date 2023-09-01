# # [Basic usage of LRP](@id docs-lrp-basics)
# This example will show you best practices for using LRP,
# building on the basics shown in the [*Getting started*](@ref docs-getting-started) section.

#md # !!! note "TLDR"
#md #
#md #     1. Use [`strip_softmax`](@ref) to strip the output softmax from your model.
#md #        Otherwise [model checks](@ref docs-lrp-model-checks) will fail.
#md #     1. Use [`canonize`](@ref) to fuse linear layers.
#md #     1. Don't just call `LRP(model)`, instead use a [`Composite`](@ref)
#md #        to apply LRP rules to your model.
#md #        Read [*Assigning rules to layers*](@ref docs-composites).
#md #     1. By default, `LRP` will call [`flatten_model`](@ref) to flatten your model.
#md #        This reduces computational overhead.

# We start out by loading a small convolutional neural network:
using ExplainableAI
using Flux

model = Chain(
    Chain(
        Conv((3, 3), 3 => 8, relu; pad=1),
        Conv((3, 3), 8 => 8, relu; pad=1),
        MaxPool((2, 2)),
        Conv((3, 3), 8 => 16; pad=1),
        BatchNorm(16, relu),
        Conv((3, 3), 16 => 8, relu; pad=1),
        BatchNorm(8, relu),
    ),
    Chain(
        Flux.flatten,
        Dense(2048 => 512, relu),
        Dropout(0.5),
        Dense(512 => 100, softmax)
    ),
);
# This model contains two chains: the convolutional layers and the fully connected layers.

# ## [Model preparation](@id docs-lrp-model-prep)
# ### [Stripping the output softmax](@id docs-lrp-strip-softmax)
# When using LRP, it is recommended to explain output logits instead of probabilities.
# This can be done by stripping the output softmax activation from the model
# using the [`strip_softmax`](@ref) function:
model = strip_softmax(model)

# If you don't remove the output softmax,
# [model checks](@ref docs-lrp-model-checks) will fail.

# ### [Canonizing the model](@id docs-lrp-canonization)
# LRP is not invariant to a model's implementation.
# Applying the [`GammaRule`](@ref) to two linear layers in a row will yield different results
# than first fusing the two layers into one linear layer and then applying the rule.
# This fusing is called "canonization" and can be done using the [`canonize`](@ref) function:
model = canonize(model)

# ### [Flattening the model](@id docs-lrp-flatten-model)
# ExplainableAI.jl's LRP implementation supports nested Flux Chains and Parallel layers.
# However, it is recommended to flatten the model before analyzing it.
#
# LRP is implemented by first running a forward pass through the model,
# keeping track of the intermediate activations, followed by a backward pass
# that computes the relevances.
#
# To keep the LRP implementation simple and maintainable,
# ExplainableAI.jl does not pre-compute "nested" activations.
# Instead, for every internal chain, a new forward pass is run to compute activations.
#
# By "flattening" a model, this overhead can be avoided.
# For this purpose, ExplainableAI.jl provides the function [`flatten_model`](@ref):
model_flat = flatten_model(model)

# This function is called by default when creating an LRP analyzer.
# Note that we pass the unflattened model to the analyzer, but `analyzer.model` is flattened:
analyzer = LRP(model)
analyzer.model

# If this flattening is not desired, it can be disabled
# by passing the keyword argument `flatten=false` to the `LRP` constructor.

# ## LRP rules
# By default, the `LRP` constructor will assign the [`ZeroRule`](@ref) to all layers.
LRP(model)

# This analyzer will return heatmaps that look identical to [`InputTimesGradient`](@ref).

# LRP's strength lies in assigning different rules to different layers,
# based on their functionality in the neural network[^1].
# ExplainableAI.jl [implements many LRP rules out of the box](@ref api-lrp-rules),
# but it is also possible to [*implement custom rules*](@ref docs-custom-rules).
#
# To assign different rules to different layers,
# use one of the [composites presets](@ref api-composite-presets),
# or create your own composite, as described in
# [*Assigning rules to layers*](@ref docs-composites).

composite = EpsilonPlusFlat() # using composite preset EpsilonPlusFlat
#-
LRP(model, composite)

# ## [Computing layerwise relevances](@id docs-lrp-layerwise)
# If you are interested in computing layerwise relevances,
# call `analyze` with an LRP analyzer and the keyword argument
# `layerwise_relevances=true`.
#
# The layerwise relevances can be accessed in the `extras` field
# of the returned [`Explanation`](@ref):
input = rand(Float32, 32, 32, 3, 1) # dummy input for our convolutional neural network

expl = analyze(input, analyzer; layerwise_relevances=true)
expl.extras.layerwise_relevances

# Note that the layerwise relevances are only kept for layers in the outermost `Chain` of the model.
# When using our unflattened model, we only obtain three layerwise relevances,
# one for each chain in the model and the output relevance:
analyzer = LRP(model; flatten=false) # use unflattened model

expl = analyze(input, analyzer; layerwise_relevances=true)
expl.extras.layerwise_relevances

# ## [Performance tips](@id docs-lrp-performance)
# ### Using LRP without a GPU
# Since ExplainableAI.jl's LRP implementation makes use of
# [Tullio.jl](https://github.com/mcabbott/Tullio.jl),
# analysis can be accelerated by loading either
# - a package from the [JuliaGPU](https://juliagpu.org) ecosystem,
#   e.g. [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl), if a GPU is available
# - [LoopVectorization.jl](https://github.com/JuliaSIMD/LoopVectorization.jl)
#   if only a CPU is available.
#
# This only requires loading the LoopVectorization.jl package before ExplainableAI.jl:
# ```julia
# using LoopVectorization
# using ExplainableAI
# ```
#
# [^1]: G. Montavon et al., [Layer-Wise Relevance Propagation: An Overview](https://link.springer.com/chapter/10.1007/978-3-030-28954-6_10)
