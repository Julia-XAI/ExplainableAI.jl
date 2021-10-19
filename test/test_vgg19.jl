using ExplainabilityMethods
using ExplainabilityMethods: ANALYZERS
using Flux
using Metalhead

using Random
Random.seed!(222)
pseudorand(dims...) = randn(MersenneTwister(123), Float32, dims...)

img = pseudorand(224, 224, 3, 1)

# Load VGG model:
# We run the reference test on the randomly intialized weights
# so we don't have to download ~550 MB on every CI run.
vgg = VGG19(; pretrain=false)
model = flatten_chain(strip_softmax(vgg.layers))

# Run analyzers
analyzers = ANALYZERS
function LRPCustom(model::Chain)
    return LRP(model, [ZBoxRule(), repeat([GammaRule()], length(model.layers) - 1)...])
end
analyzers["LRPCustom"] = LRPCustom

function approxref(ref::String, act::String)
    # Parse reference string into array and compare
    ref = eval(Meta.parse(ref))
    act = eval(Meta.parse(act))
    return isapprox(ref, act; rtol=0.03)
end

for (name, method) in analyzers
    @testset "$name" begin
        if name == "LRP"
            analyzer = method(model, ZeroRule())
        else
            analyzer = method(model)
        end
        expl, _ = analyze(img, analyzer)

        @test size(expl) == size(img)
        # Testing sum to keep reference file size lower, otherwise parser fails.
        # TODO: find a way test full (224,224,3,1) array
        @test_reference "references/vgg19/$(name).txt" sum(expl; dims=(2,3)) by = approxref
    end
end
