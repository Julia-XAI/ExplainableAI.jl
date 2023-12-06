"""
    stabilize_denom(d, [eps = 1f-9])

Replace zero terms of a matrix `d` with `eps`.
"""
function stabilize_denom(d::T, eps=T(1.0f-9)) where {T}
    iszero(d) && (return T(eps))
    return d + sign(d) * T(eps)
end
stabilize_denom(D::AbstractArray{T}, eps=T(1.0f-9)) where {T} = stabilize_denom.(D, eps)

"""
    safedivide(a, b, [eps = 1f-6])

Elementwise division of two matrices avoiding near zero terms
in the denominator by replacing them with `± eps`.
"""
function safedivide(a::AbstractArray{T}, b::AbstractArray{T}, eps=T(1.0f-9)) where {T}
    return a ./ stabilize_denom(b, T(eps))
end

"""
    drop_batch_index(I)

Drop batch dimension index (last value) from CartesianIndex.

## Example
julia> drop_batch_index(CartesianIndex(5,3,2))
CartesianIndex(5, 3)
"""
drop_batch_index(C::CartesianIndex) = CartesianIndex(C.I[1:(end - 1)])

"""
    ones_like(x)

Returns array of ones of same shape and type as `x`.

## Example
```julia-repl
julia> x = rand(Float16, 2, 4, 1)
2×4×1 Array{Float16, 3}:
[:, :, 1] =
 0.2148  0.9053   0.751    0.358
 0.38    0.09033  0.04053  0.6543

julia> ones_like(x)
2×4×1 Array{Float16, 3}:
[:, :, 1] =
 1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0
```
"""
ones_like(x::AbstractArray) = fill!(similar(x), 1)
ones_like(x::Number) = oneunit(x)

keep_positive(x::Number) = ifelse(x < 0, zero(x), x) # equivalent to relu
keep_negative(x::Number) = ifelse(x > 0, zero(x), x)
keep_positive(xs::AbstractArray) = keep_positive.(xs)
keep_negative(xs::AbstractArray) = keep_negative.(xs)

"""
    masked_copy(A, mask)

Return a copy of A on which `mask` was applied.

## Example
```julia-repl
julia> A = rand(3, 3)
3×3 Matrix{Float64}:
 0.110985  0.276119   0.660383
 0.170582  0.0315757  0.278012
 0.972022  0.18339    0.347059

julia> mask = rand(Bool, 3, 3)
3×3 Matrix{Bool}:
 1  1  1
 0  0  1
 0  1  1

julia> B = masked_copy(A, mask)
3×3 Matrix{Float64}:
 0.110985  0.276119  0.660383
 0.0       0.0       0.278012
 0.0       0.18339   0.347059
```
"""
function masked_copy(A::AbstractArray, mask::AbstractArray)
    size(A) != size(mask) && error("Size of array and mask need to match.")
    out = similar(A)
    z = zero(eltype(A))
    @inbounds for i in CartesianIndices(A)
        out[i] = ifelse(mask[i], A[i], z)
    end
    return out
end
