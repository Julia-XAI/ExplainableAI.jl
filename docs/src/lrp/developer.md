# [LRP developer documentation](@id lrp-dev-docs)
## Generic LRP rule implementation
Before we dive into package-specific implementation details 
in later sections of this developer documentation, 
we first need to cover some fundamentals of LRP, starting with the notation we use.

The generic LRP rule, of which the ``0``-, ``\epsilon``- and ``\gamma``-rules are special cases, reads[^1][^2]

```math
\begin{equation}
R_j^k = \sum_i \frac{\rho(W_{ij}) \; a_j^k}{\epsilon + \sum_{l} \rho(W_{il}) \; a_l^k + \rho(b_i)} R_i^{k+1}
\end{equation}
```

where 
*  $W$ is the weight matrix of the layer
*  $b$ is the bias vector of the layer
*  $a^k$ is the activation vector at the input of layer $k$
*  $a^{k+1}$ is the activation vector at the output of layer $k$
*  $R^k$ is the relevance vector at the input of layer $k$
*  $R^{k+1}$ is the relevance vector at the output of layer $k$
*  $\rho$ is a function that modifies parameters (what we call [`modify_parameters`](@ref docs-custom-rules-impl))
*  $\epsilon$ is a small positive constant to avoid division by zero


Subscript characters are used to index vectors and matrices 
(e.g. $b_i$ is the $i$-th entry of the bias vector), 
while the superscripts $^k$ and $^{k+1}$ 
indicate the relative positions of activations $a$ and relevances $R$ in the model.
For any $k$, $a^k$ and $R^k$ have the same shape. 

Note that every term in this equation is a scalar value,
which removes the need to differentiate between matrix and element-wise operations.

### Linear layers
LRP was developed for *deep rectifier networks*,
neural networks that are composed of linear layers with ReLU activation functions.
Linear layers are layers that can be represented as affine transformations of the form 

```math
\begin{equation}
f(x) = Wx + b \quad .
\end{equation}
```

This includes most commonly used types of layers, such as fully connected layers, 
convolutional layers, pooling layers and normalization layers.

We will now describe a generic implementation of equation (1) 
that can be applied to any linear layer.

### [The automatic differentiation fallback](@id lrp-dev-ad-fallback)
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
and replacing the activation function with the identity function.
The vector $z$ is then computed as a simple forward pass through the modified layer.
It has the same dimensionality as $R^{k+1}$ and $a^{k+1}$.

**Step 2** is a simple element-wise division of $R^{k+1}$ by $z$.
To avoid division by zero, a small constant $\epsilon$ is added to $z$ when necessary.

**Step 3** is trivial for fully connected layers, 
as $\rho(W)$ corresponds to the weight matrix of the modified layer.
For other types of linear layers however, the implementation is more involved:
A naive approach would be to construct a large matrix $W$
that corresponds to the affine transformation $Wx+b$ implemented the modified layer.
This has multiple drawbacks:
- the implementation is error-prone
- a separate implementation is required for each type of linear layer
- for some layer types, e.g. pooling layers, the matrix $W$ depends on the input
- for many layer types, e.g. convolutional layers, 
  the matrix $W$ is very large and sparse, mostly consisting of zeros,
  leading to a large computational overhead

A better approach can be found by observing that the matrix $W$ is the Jacobian
of the affine transformation $f(x) = Wx + b$.
The full vector $c$ computed in step 3 corresponds to $c = s^T W$,
a so-called Vector-Jacobian-Product (VJP) of the vector $s$ with the Jacobian $W$. 
VJPs are the building blocks of reverse-mode automatic differentiation (AD),
and efficiently implemented by most AD frameworks in a matrix-free, GPU-accelerated manner.

Functions that compute VJP's are commonly called *pullbacks*.
Using the [Zygote.jl](https://github.com/FluxML/Zygote.jl) AD system,
we obtain the output $z$ of a modified layer and its pullback `back` in a single function call:
```julia
z, back = Zygote.pullback(modified_layer, aᵏ)
```
We then call the pullback with the vector $s$ to obtain $c$:
```julia
c = back(s)
```

**Finally, step 4** consists of an element-wise multiplication of the vector $c$ 
with the input activation vector $a^k$, resulting in the relevance vector $R^k$.

This AD-based implementation is used in ExplainableAI.jl as the default method
for all layer types that don't have a more optimized implementation
(e.g. fully connected layers).
We will refer to it as the *"AD fallback"*.

For more background information on automatic differentiation, refer to the 
[JuML lecture on AD](https://adrhill.github.io/julia-ml-course/L6_Automatic_Differentiation/).


## LRP analyzer struct
The [`LRP`](@ref) analyzer struct holds three fields:
the `model` to analyze, the LRP `rules` to use and pre-allocated `modified_layers`.

As described in the section on [*Composites*](@ref docs-composites),
applying a composite to a model will return LRP rules in nested
[`ChainTuple`](@ref) and [`ParallelTuple`](@ref)s.
These wrapper types are used to match the structure of Flux models with `Chain` and `Parallel` layers while avoiding type piracy.

When creating an `LRP` analyzer with the default keyword argument `flatten=true`, 
`flatten_model` is called on the model and rules.
This is done for performance reasons, as discussed in 
[*Flattening the model*](@ref docs-lrp-flatten-model).

After running [*Model checks*](@ref docs-lrp-model-checks),
modified layers are pre-allocated, once again using the `ChainTuple` and `ParallelTuple`
wrapper types to match the structure of the model.
If a rule doesn't modify a layer, 
the corresponding entry in `modified_layers` is set to `nothing`, 
avoiding unnecessary allocations. 
If a rule requires multiple modified layers, 
the corresponding entry in `modified_layers` is set to a named tuple of modified layers.
Apart from these special cases, 
the corresponding entry in `modified_layers` is simply set to the modified layer.

For a detailed description of the layer modification mechanism, refer to the section on
[*Advanced layer modification*](@ref docs-custom-rules-advanced).

## Forward and reverse pass
When calling an `LRP` analyzer, a forward pass through the model is performed,
saving the activations $aᵏ$ for all layers $k$ in an array called `acts`.
This array is then used to pre-allocate the relevances $R^k$ for all layers in an array called `rels`.
This is possible since for any layer $k$, $a^k$ and $R^k$ have the same shape.
Finally, the last entry in `rels` is set to zeros, except for the specified output neuron, which is set to one.

We can now run the reverse pass, iterating backwards over the layers in the model
and writing the relevances $R^k$ into the pre-allocated array `rels`:

```julia
for i in length(model):-1:1
    #                  └─ loop over layers in reverse
    lrp!(rels[i], rules[i], layers[i], modified_layers[i], acts[i], rels[i+1])
    #    └─ Rᵏ: modified in-place                          └─ aᵏ    └─ Rᵏ⁺¹
end
```

This is done by calling low level functions
```julia
function lrp!(Rᵏ, rule, layer, modified_layer, aᵏ, Rᵏ⁺¹)
    Rᵏ .= ...
end
```

that implement individual LRP rules.
The correct rule is applied via 
[multiple dispatch](https://www.youtube.com/watch?v=kc9HwsxE1OY)
on the types of the arguments `rule` and `modified_layer`.
The relevance `Rᵏ` is then computed based on the input activation `aᵏ`
and the output relevance `Rᵏ⁺¹`.


The exclamation point in the function name `lrp!` is a 
[naming convention](https://docs.julialang.org/en/v1/manual/style-guide/#bang-convention)
in Julia to denote functions that modify their arguments - 
in this case the first argument `rels[i]`, which corresponds to $R^k$.

### Rule calls
As discussed in [*The AD fallback*](@ref lrp-dev-ad-fallback),
the default LRP fallback for unknown layers uses AD via 
[Zygote](https://github.com/FluxML/Zygote.jl).

Now that you are familiar with both the API and the four step computation of the generic LRP rules,
the following implementation should be straightforward to understand:
```julia
function lrp!(Rᵏ, rule, layer, modified_layer, aᵏ, Rᵏ⁺¹)
   # Use modified_layer if available
   layer = isnothing(modified_layer) ? layer : modified_layer

   ãᵏ = modify_input(rule, aᵏ)
   z, back = Zygote.pullback(modified_layer, ãᵏ)
   s = Rᵏ⁺¹ ./ modify_denominator(rule, z)
   Rᵏ .= ãᵏ .* only(back(s))
end
```

Not only `lrp!` dispatches on the rule and layer type, 
but also the internal functions `modify_input` and `modify_denominator`.
Unknown layers that are registered in the `LRP_CONFIG` use this exact function.

A good entry point into the source code is
[`/src/lrp/rules.jl`](https://github.com/adrhill/ExplainableAI.jl/blob/master/src/lrp/rules.jl).

### Specialized implementations
In other programming languages, LRP is commonly implemented in an object oriented manner,
providing a single backward pass implementation per rule.
This can be seen as a form of *single dispatch* on the rule type.

Using multiple dispatch, we can implement specialized versions of `lrp!` that not only
take into account the rule type, but also the layer type, 
for example for fully connected layers or reshaping layers. 

Reshaping layers don't affect attributions. We can therefore avoid the computational
overhead of AD by writing a specialized implementation that simply reshapes back:
```julia
function lrp!(Rᵏ, rule, layer::ReshapingLayer, modified_layer, aᵏ, Rᵏ⁺¹)
    Rᵏ .= reshape(Rᵏ⁺¹, size(aᵏ))
end
```

We can even provide a specialized implementation of the generic LRP rule for `Dense` layers. Since we can access the weight matrix directly, we can skip the use of automatic differentiation:

```math
R_j^k = \sum_i \frac{\rho(W_{ij}) \; a_j^k}{\epsilon + \sum_{l} \rho(W_{il}) \; a_l^k + \rho(b_i)} R_i^{k+1}
```

```julia
function lrp!(Rᵏ, rule, layer::Dense, modified_layer, aᵏ, Rᵏ⁺¹)
   # Use modified_layer if available
   layer = isnothing(modified_layer) ? layer : modified_layer

   ãᵏ = modify_input(rule, aᵏ)
   z = modify_denominator(rule, layer(ãᵏ))

   # Implement LRP using Einsum notation, where `b` is the batch index
   @tullio Rᵏ[j, b] = layer.weight[i, j] * ãᵏ[j, b] / z[i, b] * Rᵏ⁺¹[i, b]
end
```


For maximum low-level control beyond `modify_input` and `modify_denominator`,
you can also implement your own `lrp!` function and dispatch
on individual rule types `MyRule` and layer types `MyLayer`:
```julia
function lrp!(Rᵏ, rule::MyRule, layer::MyLayer, modified_layer, aᵏ, Rᵏ⁺¹)
    Rᵏ .= ...
end
```

[^1]: G. Montavon et al., [Layer-Wise Relevance Propagation: An Overview](https://link.springer.com/chapter/10.1007/978-3-030-28954-6_10)
[^2]: W. Samek et al., [Explaining Deep Neural Networks and Beyond: A Review of Methods and Applications](https://ieeexplore.ieee.org/document/9369420)