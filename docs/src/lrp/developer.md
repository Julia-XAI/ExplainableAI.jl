# [LRP developer documentation](@id lrp-dev-docs)
Before we dive into implementation details, 
we first need to cover the notation and theoretical fundamentals of LRP.

## Generic LRP rule
The generic LRP rule, of which the ``0``-, ``\epsilon``- and ``\gamma``-rules are special cases, reads[^1][^2]

```math
\begin{equation}
R_j^k = \sum_i \frac{\rho(W_{ij}) \; a_j^k}{\epsilon + \sum_{l} \rho(W_{il}) \; a_l^k + \rho{b_i}} R_i^{k+1}
\end{equation}
```

where 
*  $W$ is the weight matrix of the layer
*  $b$ is the bias vector of the layer
*  $a^k$ is the activation vector at the input of layer $k$
*  $a^{k+1}$ is the activation vector at the output of layer $k$
*  $R^k$ is the relevance vector at the input of layer $k$
*  $R^{k+1}$ is the relevance vector at the output of layer $k$
*  $\rho$ is a function that modifies parameters ([what we call `modify_parameters`](@ref docs-custom-rules-impl))
*  $\epsilon$ is a small positive constant to avoid division by zero


Subscript characters are used to index vectors and matrices 
(e.g. $b_i$ is the $i$-th entry of the bias vector), 
while superscripts $^k$ and $^{k+1}$ indicate the relative positions of activations $a$ and relevances $R$ in the model.
For any $k$, $a^k$ and $R^k$ have the same shape. 

Note that every term in this equation is a scalar value,
which removes the need to differentiate between matrix and element-wise operations.

### Linear layers
LRP is defined for *deep rectifier networks*,
neural networks that are composed of linear layers with ReLU activation functions.
Linear layers are layers that can be represented as affine transformations of the form 

```math
\begin{equation}
f(x) = Wx + b
\end{equation}
```

This includes most commonly used types of layers, such as fully connected layers, 
convolutional layers, pooling layers and normalization layers.

We will now describe a generic implementation of equation 1 
that can be applied to any linear layer.

###  Generic implementation using automatic differentiation
The computation of the generic LRP rule can be decomposed into four steps[^1]:
```math
\begin{array}{lr}
z_{i} = \sum_{l} \rho(W_{il}) \; a_l^k + \rho(b_i) & \text{(Step 1)} \\[0.5em]
s_{i} = R_{i}^{k+1} / (z_{i} + \epsilon)           & \text{(Step 2)} \\[0.5em]
c_{j} = \sum_i \rho(W_{ij}) \; s_{i}               & \text{(Step 3)} \\[0.5em]
R_{j}^{k} = a_{j}^{k} c_{j}                        & \text{(Step 4)}
\end{array}
```

**To compute step 1**, we first create a modified layer, 
applying $\rho$ to the weights and biases 
and replacing any activation functions with the identity function.
The vector $z$ is then computed as a simple forward pass through the modified layer.
It has the same dimensionality as $R^{k+1}$ and $a^{k+1}$.

**Step 2** is a simple element-wise division of $R^{k+1}$ by $z$.
To avoid division by zero, a small constant $\epsilon$ is added to $z$ when necessary.

**Step 3** is trivial for fully connected layers, 
as $\rho(W)$ corresponds to the weight matrix of the modified layer.
For other types of linear layers however, the implementation is more involved:
A naive approach would be to construct a large matrix $W$
that corresponds to the affine transformation $Wx+b$ implemented the layer.
This has multiple drawbacks:
- the implementation is error-prone
- a separate implementation is required for each type of linear layer
- for some layer types, e.g. pooling layers, the matrix $W$ depends on the input
- for many layer types, e.g. convolutional layers, 
  the matrix $W$ is very large and sparse, mostly consisting of zeros,
  leading to a large computational overhead

A better approach can be found by observing that the matrix $W$ is the Jacobian
of the affine transformation $f(x) = Wx + b$.
The full vector $c$ computed in step 3 corresponds to the  $c = s^T W$,
a so-called Vector-Jacobian-Product (VJP). 
VJP's are the building blocks of reverse-mode automatic differentiation (AD)[^3],
and efficiently implemented in a matrix-free, GPU-accelerated manner in most AD frameworks.

Functions that compute VJP's are commonly called *pullbacks*.
Using the [Zygote.jl](https://github.com/FluxML/Zygote.jl) AD system,
we obtain the output $z$ of a modified layer and its pullback `back` as follows:
```julia
z, back = Zygote.pullback(modified_layer, aₖ)
```
We then call the function `back` with the vector $s$ to obtain $c$.

**Finally, step 4** just consists of an element-wise multiplication of $c$ 
with the input activation $a^k$, resulting in the relevance $R^k$.

This AD-based implementation is used in ExplainableAI as the default method
for all layer types that don't have a more optimized implementation
(e.g. fully connected layers).
We will refer to it as the *"AD fallback"*.

## LRP rules in ExplainableAI.jl
The best point of entry into the source code is
[`/src/lrp/rules.jl`](https://github.com/adrhill/ExplainableAI.jl/blob/master/src/lrp/rules.jl).

Calling `analyze` on a LRP-analyzer pre-allocates modified layers by dispatching
`modify_layer` on rule and layer types. It then applies a forward-pass of the model,
keeping track of the activations `aᵏ` for each layer `k`.
The relevance `Rᵏ⁺¹` is then set to the output neuron activation and the rules are applied
in a backward-pass over the model layers and previous activations.

This is done by calling low level functions
```julia
function lrp!(Rᵏ, rule, layer, modified_layer, aᵏ, Rᵏ⁺¹)
    Rᵏ .= ...
end
```

## AD fallback
The default LRP fallback for unknown layers uses AD via [Zygote](https://github.com/FluxML/Zygote.jl).
For `lrp!`, we implement the previous four step computation using `Zygote.pullback` to
compute ``c`` from the previous equation as a VJP, pulling back ``s=R/z``:
```julia
function lrp!(Rᵏ, rule, layer, modified_layer, aᵏ, Rᵏ⁺¹)
   # Use modified_layer if available
   layer = isnothing(modified_layer) ? layer : modified_layer

   ãₖ = modify_input(rule, aᵏ)
   z, back = Zygote.pullback(modified_layer, ãₖ)
   s = Rᵏ⁺¹ ./ modify_denominator(rule, z)
   Rᵏ .= ãₖ .* only(back(s))
end
```

You can see how `modify_input` and `modify_denominator` dispatch on rule and layer types.
Unknown layers that are registered in the `LRP_CONFIG` use this exact function.

These functions in-place modify a pre-allocated array of the input relevance `Rᵏ`.
(The `!` is a [naming convention](https://docs.julialang.org/en/v1/manual/style-guide/#bang-convention)
in Julia to denote functions that modify their arguments.)

The correct rule is applied via [multiple dispatch](https://www.youtube.com/watch?v=kc9HwsxE1OY)
on the types of the arguments `rule` and `modified_layer`.
The relevance `Rᵏ` is then computed based on the input activation `aᵏ`
and the output relevance `Rᵏ⁺¹`.
Multiple dispatch is also used to dispatch `modify_parameters` and `modify_denominator`
on the rule and layer type.

## Specialized implementations
We can also implement specialized versions of `lrp!` based on the type of `layer`,
e.g. reshaping layers.

Reshaping layers don't affect attributions. We can therefore avoid the computational
overhead of AD by writing a specialized implementation that simply reshapes back:
```julia
function lrp!(Rᵏ, rule, _layer::ReshapingLayer, _modified_layer, aᵏ, Rᵏ⁺¹)
    Rᵏ .= reshape(Rᵏ⁺¹, size(aᵏ))
end
```

Since the rule type didn't matter in this case, we didn't specify it.

We can even implement the generic rule as a specialized implementation for `Dense` layers, since

```math
R_j^k = \sum_i \frac{W_{ij} a_j^k}{\sum_{l} W_{il} a_l^k + b_i} R_i^{k+1}
```

```julia
function lrp!(Rᵏ, rule, layer::Dense, modified_layer, aᵏ, Rᵏ⁺¹)
   # Use modified_layer if available
   layer = isnothing(modified_layer) ? layer : modified_layer

   ãₖ = modify_input(rule, aᵏ)
   z = modify_denominator(rule, layer(ãₖ))

   # Implement LRP using Einsum notation, where `b` is the batch index
   @tullio Rᵏ[j, b] = layer.weight[i, j] * ãₖ[j, b] / z[i, b] * Rᵏ⁺¹[i, b]
end
```


For maximum low-level control beyond `modify_input` and `modify_denominator`,
you can also implement your own `lrp!` function and dispatch
on individual rule types `MyRule` and layer types `MyLayer`:
```julia
function lrp!(Rᵏ, rule::MyRule, layer::MyLayer, _modified_layer, aᵏ, Rᵏ⁺¹)
    Rᵏ .= ...
end
```

[^1]: G. Montavon et al., [Layer-Wise Relevance Propagation: An Overview](https://link.springer.com/chapter/10.1007/978-3-030-28954-6_10)
[^2]: W. Samek et al., [Explaining Deep Neural Networks and Beyond: A Review of Methods and Applications](https://ieeexplore.ieee.org/document/9369420)
[^3]: For more background information, refer to the [JuML lecture on automatic differentiation](https://adrhill.github.io/julia-ml-course/L6_Automatic_Differentiation/).