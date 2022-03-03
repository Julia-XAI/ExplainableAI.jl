using ExplainabilityMethods
using Flux
using Metalhead
using Metalhead: weights

vgg = VGG19()
Flux.loadparams!(vgg, Metalhead.weights("vgg19"))

model = strip_softmax(vgg.layers);

using Images
using TestImages

img = testimage("chelsea")

using DataAugmentation

# Coefficients taken from PyTorch's ImageNet normalization code
μ = [0.485, 0.456, 0.406]
σ = [0.229, 0.224, 0.225]
transform = CenterResizeCrop((224, 224)) |> ImageToTensor() |> Normalize(μ, σ)

item = Image(img)
input = apply(transform, item) |> itemdata
input = permutedims(input, (2,1,3))[:,:,:,:] * 255; # flip X/Y axes, add batch dim. and rescale

analyzer = LRPZero(model)
expl, out = analyze(input, analyzer);

heatmap(expl)

model = flatten_model(model)

rules = [
    ZBoxRule(),
    repeat([GammaRule()], 15)...,
    repeat([ZeroRule()], length(model) - 16)...
]

analyzer = LRP(model, rules)
expl, out = analyze(input, analyzer)
heatmap(expl)

struct MyCustomLRPRule <: AbstractLRPRule end

function modify_params(::MyCustomLRPRule, W, b)
    ρW = W + 0.1 * relu.(W)
    return ρW, b
end

analyzer = LRP(model, MyCustomLRPRule())
expl, out = analyze(input, analyzer)
heatmap(expl)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

