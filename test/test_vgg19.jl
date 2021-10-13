using ExplainabilityMethods
using Flux
using Metalhead

using Images
using ImageCore
using ImageMagick
using ImageInTerminal

# Load test image and use preprocess it
include("./preprocessing.jl")
imgp = preprocess("./img/leuchtturm2.jpg")

# Load VGG model
vgg = VGG19()
model = strip_softmax(vgg.layers)

# Run analyzers
analyzer1 = Gradient(model)
expl, _ = analyze(imgp, analyzer1)
h = heatmap(expl)
save("Gradient.png", h)
imshow(h)

analyzer2 = InputTimesGradient(model)
expl, _ = analyze(imgp, analyzer2)
h = heatmap(expl)
save("InputTimesGradient.png", h)
imshow(h)

analyzer3 = LRPZero(model)
expl, _ = analyze(imgp, analyzer3)
h = heatmap(expl)
save("LRPZero.png", h)
imshow(h)

analyzer4 = LRPEpsilon(model)
expl, _ = analyze(imgp, analyzer4)
h = heatmap(expl)
save("LRPEpsilon.png", h)
imshow(h)

analyzer5 = LRPGamma(model)
expl, _ = analyze(imgp, analyzer5)
h = heatmap(expl)
save("LRPGamma.png", h)
imshow(h)

rules = [ZBoxRule(), repeat([GammaRule()], length(model.layers) - 1)...]
analyzer6 = LRP(model, rules)
expl, _ = analyze(imgp, analyzer6)
h = heatmap(expl)
save("LRPCustom.png", h)
imshow(h)

# ## Compare analyzers
# fig = compare(imgp, labels, [analyzer1, analyzer2])
# save("test/testgrid.png", fig)
