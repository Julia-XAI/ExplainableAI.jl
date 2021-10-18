using ExplainabilityMethods
using ExplainabilityMethods: SumReducer

# Defaults assume nchannels=3
A = rand(Float32, 3, 4, 5)
B = reshape(A, 3, 1, 4, 5, 1)
@test heatmap(A) ≈ heatmap(B)

A = rand(Float32, 3, 4, 5)
B = reshape(A, 3, 1, 4, 1, 5)
@test heatmap(A; reducer=SumReducer()) ≈ heatmap(B; reducer=SumReducer())

# Test with single channel
A = rand(Float32, 4, 5)
B = reshape(A, 4, 1, 1, 5, 1)
@test heatmap(A; nchannels=1) ≈ heatmap(B; nchannels=1)

# Test with 2 color channels
A = rand(Float32, 2, 3, 4)
B = reshape(A, 2, 1, 1, 3, 4)
@test heatmap(A; nchannels=2) ≈ heatmap(B; nchannels=2)
