using ExplainabilityMethods
using ExplainabilityMethods: ANALYZERS
using Flux
using JLD2

using Random
pseudorand(T, dims...) = rand(MersenneTwister(123), T, dims...)
img = pseudorand(Float32, (224, 224, 3, 1))

# Load VGG model:
# We run the reference test on the randomly intialized weights
# so we don't have to download ~550 MB on every CI run.
include("./vgg19.jl")
vgg19 = VGG19(; pretrain=false)
model = flatten_chain(strip_softmax(vgg19.layers))

# Run analyzers
analyzers = ANALYZERS
function LRPCustom(model::Chain)
    return LRP(model, [ZBoxRule(), repeat([GammaRule()], length(model.layers) - 1)...])
end
analyzers["LRPCustom"] = LRPCustom

for (name, method) in analyzers
    @time @testset "$name" begin
        print("Timing $name on VGG19...")
        if name == "LRP"
            analyzer = method(model, ZeroRule())
        else
            analyzer = method(model)
        end
        expl, _ = analyze(img, analyzer)

        @test size(expl) == size(img)
        @test_reference "references/vgg19/$(name).jld2" Dict("expl" => expl) by =
            (r, a) -> isapprox(r["expl"], a["expl"]; rtol=0.05)
    end
end
