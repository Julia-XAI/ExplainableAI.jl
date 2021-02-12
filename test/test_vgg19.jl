using ImageMagick # jpg I/O
using Flux
using Metalhead
# Load useful things from Metalhead.utils
using Metalhead: classify, forward, labels, load_img
using PrettyTables

vgg = VGG19()

# ## Print layers in VGG19
println("VGG-19 layers:")
for layer in vgg.layers.layers
    println("\t", layer)
end

# ## Load test image
img = load("./test/img/elephant.jpg")

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
    ids = sortperm(scores;rev=true)[1:n] # indices of Top n classes

    # Draw table
    data = hcat(labs[ids], probs[ids])
    header = ["Label" "Score [%]"]
    pretty_table(data, header)
    return nothing
end

print_scores(vgg, img)
