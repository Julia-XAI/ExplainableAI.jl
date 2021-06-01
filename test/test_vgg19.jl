using ExplainabilityMethods
using Flux
using Metalhead
using Metalhead: classify, forward, labels, load_img

using ImageMagick # jpg I/O
using PrettyTables

vgg = VGG19()

# ## Print layers in VGG19
println("VGG-19 layers:")
for (i, layer) in enumerate(vgg.layers.layers)
    println("\t Layer $(i): ", layer)
    # println("\t\t", size.(Flux.params(layer))) # access params
end

println("Param sizes: $(size.(Flux.params(vgg)))")

# ## Load test image
img = load("./test/img/onion.jpg")

# ## String output
# This is equivalent to
# ```julia
# classify(model::ClassificationModel, im) = Flux.onecold(forward(model, load_img(im)), labels(model))
# ```
classify(vgg, img)

# Print sorted scores
function print_scores(model, img; n=10)
    probs = forward(model, img) * 1f2 # class probabilities
    labs = labels(model)
    ids = sortperm(probs; rev=true)[1:n] # indices of Top n classes

    # Draw table
    data = hcat(labs[ids], probs[ids])
    header = ["Label" "Score [%]"]
    pretty_table(data, header)
    return nothing
end

print_scores(vgg, img)

## Test LRP
using Random
using LayerwiseRelevancePropagation: _LRP
rng = MersenneTwister(1234)

dense_test = vgg.layers[27]
W_test = Flux.params(dense_test)[1]
isize = (4096,) # input dimension
osize = Flux.outdims(dense_test, isize)

a_in = randn(rng, Float32, isize)
R_out = randn(rng, Float32, osize)

println("Asdf:")
println.(size.([a_in, W_test]))
println("mult:", size(W_test * a_in))
test = permutedims(a_in) * W_test'ᵀ
println("mult:", size(test))

R_in = LRP_generic(dense_test, a_in, R_out)


Upper
