using ExplainabilityMethods
using ExplainabilityMethods: MaxAbsNormalizer, RangeNormalizer
using ExplainabilityMethods: MaxAbsReducer, SumReducer

# Defaults assume nchannels=3
A = rand(Float32, 3, 4, 5)
B = reshape(A, 3, 1, 4, 5, 1)
@test heatmap(A; normalizer=MaxAbsNormalizer(), reducer=MaxAbsReducer()) ≈
    heatmap(B; normalizer=MaxAbsNormalizer(), reducer=MaxAbsReducer())
@test heatmap(A; normalizer=RangeNormalizer(), reducer=SumReducer()) ≈
    heatmap(B; normalizer=RangeNormalizer(), reducer=SumReducer())

A = rand(Float32, 3, 4, 5)
B = reshape(A, 3, 1, 4, 1, 5)
@test heatmap(A; normalizer=MaxAbsNormalizer(), reducer=MaxAbsReducer()) ≈
    heatmap(B; normalizer=MaxAbsNormalizer(), reducer=MaxAbsReducer())
@test heatmap(A; normalizer=RangeNormalizer(), reducer=SumReducer()) ≈
    heatmap(B; normalizer=RangeNormalizer(), reducer=SumReducer())

# Test with single channel
A = rand(Float32, 4, 5)
B = reshape(A, 4, 1, 1, 5, 1)
@test heatmap(A; nchannels=1, normalizer=MaxAbsNormalizer(), reducer=MaxAbsReducer()) ≈
    heatmap(B; nchannels=1, normalizer=MaxAbsNormalizer(), reducer=MaxAbsReducer())
@test heatmap(A; nchannels=1, normalizer=RangeNormalizer(), reducer=SumReducer()) ≈
    heatmap(B; nchannels=1, normalizer=RangeNormalizer(), reducer=SumReducer())

# Test with 2 color channels
A = rand(Float32, 2, 3, 4)
B = reshape(A, 2, 1, 1, 3, 4)
@test heatmap(A; nchannels=2, normalizer=MaxAbsNormalizer(), reducer=MaxAbsReducer()) ≈
    heatmap(B; nchannels=2, normalizer=MaxAbsNormalizer(), reducer=MaxAbsReducer())
@test heatmap(A; nchannels=2, normalizer=RangeNormalizer(), reducer=SumReducer()) ≈
    heatmap(B; nchannels=2, normalizer=RangeNormalizer(), reducer=SumReducer())
