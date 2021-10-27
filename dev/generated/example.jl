using ExplainabilityMethods
using Flux
using Metalhead
using Metalhead: weights

vgg = VGG19()
Flux.loadparams!(vgg, Metalhead.weights("vgg19"))

model = strip_softmax(vgg.layers)

using Images
using TestImages

img_raw = testimage("chelsea")

include("../utils/preprocessing.jl")
img = preprocess(img_raw)
size(img)

analyzer = LRPZero(model)
expl, out = analyze(img, analyzer);

heatmap(expl)

model = flatten_chain(model)

rules = [
    ZBoxRule(), repeat([GammaRule()], 15)..., repeat([ZeroRule()], length(model) - 16)...
]

analyzer = LRP(model, rules)
expl, out = analyze(img, analyzer)
heatmap(expl)

struct MyCustomLRPRule <: AbstractLRPRule end

function modify_params(::MyCustomLRPRule, W, b)
    ρW = W + 0.1 * relu.(W)
    return ρW, b
end

analyzer = LRP(model, MyCustomLRPRule())
expl, out = analyze(img, analyzer)
heatmap(expl)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

