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
batch_dim_view(A::AbstractArray{T,N}) where {T,N} = view(A, ntuple(_ -> :, Val(N + 1))...)

"""
    drop_batch_dim(I)

Drop batch dimension index (last value) from CartesianIndex.

## Example
julia> drop_batch_dim(CartesianIndex(5,3,2))
CartesianIndex(5, 3)
"""
drop_batch_dim(C::CartesianIndex) = CartesianIndex(C.I[1:(end - 1)])

# Utils for printing model check summary using PrettyTable.jl
_print_name(layer) = "$layer"
_print_name(layer::Parallel) = "Parallel(...)"
_print_activation(layer) = hasproperty(layer, :σ) ? "$(layer.σ)" : "—"
_print_activation(layer::Parallel) = "—"

function _show_check_summary(
    c::Chain, layer_names, layer_checks, activation_names, activation_checks
)
    hl_pass = Highlighter((data, i, j) -> j in (3, 5) && data[i, j]; foreground=:green)
    hl_fail = Highlighter((data, i, j) -> j in (3, 5) && !data[i, j]; foreground=:red)
    data = hcat(
        collect(1:length(c)),
        layer_names,
        collect(layer_checks),
        activation_names,
        collect(activation_checks),
    )
    pretty_table(
        data;
        header=["", "Layer", "Layer supported", "Activation", "Act. supported"],
        alignment=[:r, :l, :r, :c, :r],
        highlighters=(hl_pass, hl_fail),
    )
    return nothing
end
