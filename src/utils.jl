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
