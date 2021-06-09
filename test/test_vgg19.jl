using ExplainabilityMethods
using Flux
using Metalhead

using ImageCore
using ImageMagick
using ImageInTerminal

# ## Load test image
# And use `preprocess` to convert from CHW (Image.jl's channel ordering) to WHCN (Flux.jl's ordering)
# and enforce Float32, as that seems important to Flux
img = RGB.(load("./img/maltese.jpg"))
imgp = Metalhead.preprocess(img)

# ## Load VGG model
vgg = VGG19()
chain = vgg.layers
labels = Metalhead.labels(vgg)

# model_summary(chain)

# ## Run analyzers
analyzer = Gradient(chain)
class, expl = classify_and_explain(imgp, labels, analyzer)
imshow(heatmap(expl))

analyzer = InputTimesGradient(chain)
class, expl = classify_and_explain(imgp, labels, analyzer)
imshow(heatmap(expl))
