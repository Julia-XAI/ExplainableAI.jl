
"""
    augment_batch_dim(input, n)

Repeat each sample in input batch n-times along batch dimension.
This turns arrays of size `(..., B)` into arrays of size `(..., B*n)`.

## Example
```julia-repl
julia> A = [1 2; 3 4]
2×2 Matrix{Int64}:
 1  2
 3  4

julia> augment_batch_dim(A, 3)
2×6 Matrix{Int64}:
 1  1  1  2  2  2
 3  3  3  4  4  4
```
"""
function augment_batch_dim(input::AbstractArray{T,N}, n) where {T,N}
    return repeat(input; inner=(ntuple(_ -> 1, Val(N - 1))..., n))
end

"""
    reduce_augmentation(augmented_input, n)

Reduce augmented input batch by averaging the explanation for each augmented sample.
"""
function reduce_augmentation(input::AbstractArray{T,N}, n) where {T<:AbstractFloat,N}
    return cat(
        (
            Iterators.map(1:n:size(input, N)) do i
                augmentation_range = ntuple(_ -> :, Val(N - 1))..., i:(i + n - 1)
                sum(view(input, augmentation_range...); dims=N) / n
            end
        )...; dims=N
    )::Array{T,N}
end
"""
    augment_indices(indices, n)

Strip batch indices and return inidices for batch augmented by n samples.

## Example
```julia-repl
julia> inds = [CartesianIndex(5,1), CartesianIndex(3,2)]
2-element Vector{CartesianIndex{2}}:
 CartesianIndex(5, 1)
 CartesianIndex(3, 2)

julia> augment_indices(inds, 3)
6-element Vector{CartesianIndex{2}}:
 CartesianIndex(5, 1)
 CartesianIndex(5, 2)
 CartesianIndex(5, 3)
 CartesianIndex(3, 4)
 CartesianIndex(3, 5)
 CartesianIndex(3, 6)
```
"""
function augment_indices(inds::Vector{CartesianIndex{N}}, n) where {N}
    indices_wo_batch = [i.I[1:(end - 1)] for i in inds]
    return map(enumerate(repeat(indices_wo_batch; inner=n))) do (i, idx)
        CartesianIndex{N}(idx..., i)
    end
end
