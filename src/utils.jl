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
    batch_dim_view(A)

Return a view onto the array `A` that contains an extra singleton batch dimension at the end.
This avoids allocating a new array.

## Example
```juliarepl
julia> A = [1 2; 3 4]
2×2 Matrix{Int64}:
 1  2
 3  4

julia> batch_dim_view(A)
2×2×1 view(::Array{Int64, 3}, 1:2, 1:2, :) with eltype Int64:
[:, :, 1] =
 1  2
 3  4
```
"""
batch_dim_view(A::AbstractArray{T,N}) where {T,N} = view(A, ntuple(Returns(:), N + 1)...)

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
ones_like(x::AbstractArray) = ones(eltype(x), size(x))
ones_like(x::Number) = oneunit(x)

function keep_positive!(x::AbstractArray{T}) where {T}
    z = zero(T)
    x[x .< 0] .= z
    return x
end
function keep_negative!(x::AbstractArray{T}) where {T}
    z = zero(T)
    x[x .> 0] .= z
    return x
end
keep_positive(x) = keep_positive!(deepcopy(x))
keep_negative(x) = keep_negative!(deepcopy(x))
