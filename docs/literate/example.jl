# # Getting started
# ## Preparing the model
# ExplainabilityMethods.jl can be used on any classifier.
# In this tutorial we will be using a pretrained VGG-19 model from
# [Metalhead.jl](https://github.com/FluxML/Metalhead.jl)
using ExplainabilityMethods
using Flux
using Metalhead
using Metalhead: weights

vgg = VGG19()
Flux.loadparams!(vgg, Metalhead.weights("vgg19"))

#md # !!! note "Pretrained weights"
#md #     This doc page was generated using Metalhead `v0.6.0`.
#md #     At the time you read this, Metalhead might already have implemented weight loading
#md #     via `VGG19(; pretrain=true)`, in which case `loadparams!` is not necessary.

# In case they exist, we need to strip softmax activations from the output before analyzing:
model = strip_softmax(vgg.layers)

# We also need to load an image
using Images
using TestImages

img_raw = testimage("chelsea")

# which we preprocess for VGG-19
include("../utils/preprocessing.jl")
img = preprocess(img_raw)
size(img)

# ## Calling the analyzer
# We can now select an analyzer of our choice
# and call `analyze` to get an explaination `expl`:
analyzer = LRPZero(model)
expl, out = analyze(img, analyzer);

#md # !!! note "Neuron selection"
#md #     To get an explaination with respect to a specific output neuron (e.g. class 42) call
#md #     ```julia
#md #     expl, out = analyze(img, analyzer, 42)
#md #     ```
#
# Finally, we can visualize the explaination through heatmapping:
heatmap(expl)

# Currently, the following analyzers are implemented:
#
# ```
# ├── Gradient
# ├── InputTimesGradient
# └── LRP
#     ├── LRPZero
#     ├── LRPEpsilon
#     └── LRPGamma
# ```

# ## Custom rules composites
# If our model is a "flat" chain of Flux layers, we can assign LRP rules
# to each layer individually. For this purpose,
# ExplainabilityMethods exports the method `flatten_chain`:
model = flatten_chain(model)

#md # !!! note "Flattening models"
#md #     Not all models can be flattened, e.g. those using
#md #     `Parallel` and `SkipConnection` layers.
#
# Now we set a rule for each layer
rules = [
    ZBoxRule(), repeat([GammaRule()], 15)..., repeat([ZeroRule()], length(model) - 16)...
]
# to define a custom LRP analyzer:
analyzer = LRP(model, rules)
expl, out = analyze(img, analyzer)
heatmap(expl)

# ## Custom rules
# Let's define a rule that modifies the weights and biases of our layer on the forward pass.
# The rule has to be of type `AbstractLRPRule`.
struct MyCustomLRPRule <: AbstractLRPRule end

# It is then possible to dispatch on the utility functions `modify_layer`, `modify_params`
# and `modify_denominator` with our rule type `MyCustomLRPRule`
# to define custom rules without writing boilerplate code.
function modify_params(::MyCustomLRPRule, W, b)
    ρW = W + 0.1 * relu.(W)
    return ρW, b
end

# We can directly use this rule to make an analyzer!
analyzer = LRP(model, MyCustomLRPRule())
expl, out = analyze(img, analyzer)
heatmap(expl)

#md # !!! note "PRs welcome"
#md #     If you implement a rule that's not included in ExplainabilityMethods, please make a PR to
#md #     [`src/lrp_rules.jl`](https://github.com/adrhill/ExplainabilityMethods.jl/blob/master/src/lrp_rules.jl)!
