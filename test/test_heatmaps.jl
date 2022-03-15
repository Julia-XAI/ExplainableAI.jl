using ExplainabilityMethods

# NOTE: Heatmapping assumes Flux's WHCN convention (width, height, color channels, batch size).
shape = (2, 2, 3, 1)
A = reshape(collect(Float32, 1:prod(shape)), shape)

reducers = [:sum, :maxabs, :norm]
normalizers = [:extrema, :centered]
for r in reducers
    for n in normalizers
        h = heatmap(A; reduce=r, normalize=n)
        @test_reference "references/heatmaps/reduce_$(r)_normalize_$(n).txt" h
    end
end

@test_throws ArgumentError heatmap(A, reduce=:foo)
@test_throws ArgumentError heatmap(A, normalize=:bar)

B = reshape(A, 2, 2, 3, 1, 1)
@test_throws DomainError heatmap(B)
B = reshape(A, 2, 2, 3)
@test_throws DomainError heatmap(B)
B = reshape(A, 2, 2, 1, 3)
@test_throws DomainError heatmap(B)
