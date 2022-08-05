using ExplainableAI
using Flux

# Load VGG model:
# We run the reference test on the randomly intialized weights
# so we don't have to download ~550 MB on every CI run.
include("./vgg11.jl")
model = VGG11(; pretrain=false)
model = flatten_model(strip_softmax(model.layers))

# This composite is non-sensical, but covers as many composite primitives as possible
composite = Composite(
    ZeroRule(), # default rule
    GlobalRule(PassRule()), # override default rule
    RuleMap(
        Dict(
            ConvLayer => AlphaBetaRule(2.0f0, 1.0f0),
            Dense => EpsilonRule(1.0f-6),
            PoolingLayer => EpsilonRule(1.0f-6),
        ),
    ),
    FirstNRuleMap(7, Dict(Conv => FlatRule())),
    LastNRuleMap(3, Dict(Dense => EpsilonRule(1.0f-7))),
    RangeRuleMap(4:10, Dict(PoolingLayer => EpsilonRule(1.0f-5))),
    LayerRule(9, AlphaBetaRule(1.0f0, 0.0f0)),
    FirstRule(ZBoxRule(-3.0f0, 3.0f0)),
    LastRule(ZeroRule()),
)
analyzer = LRP(model, composite)
@test analyzer.rules == [
    ZBoxRule(-3.0f0, 3.0f0)
    EpsilonRule(1.0f-6)
    FlatRule()
    EpsilonRule(1.0f-5)
    FlatRule()
    FlatRule()
    EpsilonRule(1.0f-5)
    AlphaBetaRule(2.0f0, 1.0f0)
    AlphaBetaRule(1.0f0, 0.0f0)
    EpsilonRule(1.0f-5)
    AlphaBetaRule(2.0f0, 1.0f0)
    AlphaBetaRule(2.0f0, 1.0f0)
    EpsilonRule(1.0f-6)
    PassRule()
    EpsilonRule(1.0f-6)
    PassRule()
    EpsilonRule(1.0f-7)
    PassRule()
    ZeroRule()
]
