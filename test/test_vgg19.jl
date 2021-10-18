using ExplainabilityMethods
using ExplainabilityMethods: ANALYZERS
using Flux
using Metalhead

using Images
using ImageMagick
using TestImages
using Random
Random.seed!(222)

include("./vgg_preprocessing.jl")

# Load test image and use preprocess it
img = testimage("monarch_color")
imgp = preprocess(img)

# Load VGG model
vgg = VGG19()
model = flatten_chain(strip_softmax(vgg.layers))

# Run analyzers
analyzers = ANALYZERS
function LRPCustom(model::Chain)
    return LRP(model, [ZBoxRule(), repeat([GammaRule()], length(model.layers) - 1)...])
end
analyzers["LRPCustom"] = LRPCustom

for (name, method) in analyzers
    if name == "LRP"
        analyzer = method(model, ZeroRule())
    else
        analyzer = method(model)
    end
    expl, _ = analyze(imgp, analyzer)

    # Since Zygote gradients are not deterministic and ReferenceTests is best suited for images,
    # we compare if the heatmaps are approximately the same:
    h = heatmap(expl)
    @test_reference "references/vgg19/$(name).png" h by =
        (ref, x) -> isapprox(ref, x; atol=0.05)
end
