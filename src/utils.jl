"""
    stabilize_denom(d; eps = 1f-6)

Replace zero terms of a matrix `d` with `eps`.
"""
stabilize_denom(d::Real; eps=1.0f-9) = ifelse(d ≈ 0, d + sign(d) * eps, d)
function stabilize_denom(d::AbstractArray; eps=1.0f-9)
    return d + ((d .≈ 0) + sign.(d)) * eps
end

"""
    safedivide(a, b; eps = 1f-6)

Elementwise division of two matrices avoiding near zero terms
in the denominator by replacing them with `± eps`.
"""
safedivide(a, b; eps=1.0f-9) = a ./ stabilize_denom(b; eps=eps)

"""
    drop_singleton_dims(a)

Drop dimensions of size 1 from array.
"""
function drop_singleton_dims(a::AbstractArray)
    return dropdims(a; dims=tuple(findall(size(a) .== 1)...))
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
