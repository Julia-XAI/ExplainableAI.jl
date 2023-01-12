using ExplainableAI: ChainTuple, ParallelTuple
using ExplainableAI: head_tail, chainmap

# Test head_tail
@test head_tail(1, 2, 3, 4) == (1, (2, 3, 4))
@test head_tail((1, 2, 3, 4)) == (1, (2, 3, 4))
@test head_tail([1, 2, 3, 4]) == (1, (2, 3, 4))
@test head_tail(1, (2, 3), 4) == (1, ((2, 3), 4))
@test head_tail(1) == (1, ())
@test head_tail() == ()

# Test chainmap
c = Chain(
    Chain(Dense(4, 4, relu), Dense(4, 10)),
    Parallel(+, Dense(10, 2, relu), Dense(10, 2, selu), Chain(Dense(10, 5, sigmoid), Dense(5, 2, gelu))),
    Dense(2, 1, leakyrelu),
)
shapes = chainmap(l -> size(l.weight), c)
@test shapes == ChainTuple(
    ChainTuple((4, 4), (10, 4)),
    ParallelTuple((2, 10), (2, 10), ChainTuple((5, 10), (2, 5))),
    (1, 2),
)
