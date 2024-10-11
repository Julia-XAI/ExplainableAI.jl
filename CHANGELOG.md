# ExplainableAI.jl

## Version `v0.9.0`
- ![Feature][badge-feature] Support selection of AD backends via DifferentiationInterface.jl. ([#167])
- ![BREAKING][badge-breaking] For gradient-based XAI methods, an AD backend must now be manually loaded. 
  To keep using the default Zygote backend, this simply requires adding `using Zygote` to your code. ([#177]) 
- ![BREAKING][badge-breaking] `Gradient`, `InputTimesGradient` and `GradCAM` analyzers now have an additional `backend` field and type parameter. ([#167])
- ![BREAKING][badge-breaking] Update XAIBase interface to `v4`. 
  This adds a field to the `Explanation` return type and removes the `add_batch_dim` keyword argument.
  Refer to the [XAIBase.jl changelog](https://github.com/Julia-XAI/XAIBase.jl/blob/main/CHANGELOG.md#version-v400) for more information. ([#174])


## Version `v0.8.0`
This release removes the automatic reexport of heatmapping functionality.
Users are now required to manually load 
[VisionHeatmaps.jl][VisionHeatmaps] and/or [TextHeatmaps.jl][TextHeatmaps].

This reduces the maintenance burden for new heatmapping features 
and the amount of dependencies for users who don't need heatmapping functionality.

- ![BREAKING][badge-breaking] Removed reexport of heatmapping functionality by updating XAIBase dependency to `v3.0.0` ([#162]).
- ![Feature][badge-feature] Added `GradCAM` analyzer ([#155]). Try it with [VisionHeatmaps.jl][VisionHeatmaps]'s new `heatmap_overlay` feature.

## Version `v0.7.0`
This release moves the core interface (`Explanation`, `heatmap`, `analyze`) 
into a separate package called [XAIBase.jl](https://github.com/Julia-XAI/XAIBase.jl).
Developers can make use of the [XAIBase.jl interface](https://julia-xai.github.io/XAIDocs/XAIBase/)
to quickly implement or prototype new methods without having to write boilerplate code.

As announced with version `v0.6.2`, this is first release without LRP, which has been moved to a separate package called [RelevancePropagation.jl](https://github.com/Julia-XAI/RelevancePropagation.jl). This separation is enabled by the new common XAIBase.jl interface.

- ![BREAKING][badge-breaking] Move core interface into XAIBase.jl package ([#154]).
  - Renamed `Explanation` field `neuron_selection` to `output_selection`
  - Added `Explanation` field `heatmap` for heatmapping presets
- ![BREAKING][badge-breaking] Move LRP into RelevancePropagation.jl ([#157])
- ![BREAKING][badge-breaking] Remove ImageNet preprocessing utilities ([#159])
- ![Documentation][badge-docs] Partially move documentation into the [Julia-XAI ecosystem documentation](https://github.com/Julia-XAI/XAIDocs)

## Version `v0.6.3`
- ![Enhancement][badge-enhancement] Allow Gradient analyzers on non-Flux models ([#150])
- ![Bugfix][badge-bugfix] Fix typo in `BATCHDIM_MISSING` error ([#150])

## Version `v0.6.2`
This is first release of ExplainableAI.jl as part of the 
[Julia-XAI](https://github.com/Julia-XAI) organization ([#149])
and the last minor release that includes LRP before it is moved to its own separate package.

- ![Feature][badge-feature] Add Concept Relevance Propagation analyzer `CRP` ([#146], [#148])
- ![Feature][badge-feature] Add option to process heatmaps batch-wise 
  using keyword argument `process_batch=true` ([#146], [#148])
- ![Bugfix][badge-bugfix] Remove `FlatRule` on dense layers 
  from `EpsilonPlusFlat` and `EpsilonAlpha2Beta1Flat` composite presets ([#147])

## Version `v0.6.1`
This release brings GPU support to all analyzers.
- ![Feature][badge-feature] Support LRP on GPUs ([#142], [#140])
- ![Feature][badge-feature] Support gradient analyzers on GPUs ([#144])
- ![Enhancement][badge-enhancement] Make Tullio optional dependency using package extensions ([#141])
- ![Documentation][badge-docs] Document GPU support ([#145])

## Version `v0.6.0`
This release brings a large refactor of LRP analyzers, 
supporting nested "dataflow layers" from Flux.jl like `Chain` and `Parallel` layers. 
This enables LRP on more complex model architectures like ResNets.

Due to the fact that these new features require a breaking release,
we've used the occasion to clean up the API. 
Since the number of changes is large, this changelog has been 
split between changes to LRP analyzers and more general changes to the package.

### Changes to LRP analyzers
Breaking changes:
- ![BREAKING][badge-breaking] Remove all unicode characters from user-facing API ([#107])
    - `EpsilonRule`: argument `epsilon` replaces `ϵ` 
    - `GammaRule`: argument `gamma` replaces `γ` 
    - `AlphaBetaRule`: arguments `alpha` and `beta` replace `α`, `β` 
- ![BREAKING][badge-breaking] Rename `LRP` analyzer keyword argument `is_flat=false` to `flatten=true` ([#119])
- ![BREAKING][badge-breaking] Remove `check_model`, replaced by non-exported `check_lrp_compat` ([#119])
- ![BREAKING][badge-breaking] Replace `layerwise_relevances` field of `Explanation` return type by optional named tuple `extras`.
    Access layerwise relevances via `extras.layerwise_relevances`. ([#126])
- ![BREAKING][badge-breaking] Remove composite `LastNTypeRule` ([#119]) 
- ![BREAKING][badge-breaking] Rename composite primitives to avoid confusion with LRP rules ([#130])
    - rename `*Rule` to `*Map`
    - rename `*TypeRule` to `*TypeMap`

Breaking changes to commonly extended internal functions:
- ![BREAKING][badge-breaking] Internal `lrp!` rule calls require extra argument `layer` ([#119])
- ![BREAKING][badge-breaking] Pre-allocate modified layers, replacing `modify_param!` with `modify_parameters` ([#102])

New features and enhancements:
- ![Feature][badge-feature] Support nested Flux `Chain`s ([#119])
- ![Feature][badge-feature] Support `Parallel` layers ([#135], [#138])
- ![Feature][badge-feature] Support `BatchNorm` layers ([#129], [#134])
- ![Feature][badge-feature] Add `GeneralizedGammaRule` ([#109])
- ![Feature][badge-feature] Support nested indexing in composite primitive `LayerMap` ([#131])
- ![Enhancement][badge-enhancement] Pre-allocate modified layers in `LRP` analyzer field `modified_layers` ([#119])
- ![Enhancement][badge-enhancement] Set LRP output relevance to one ([#128])
- ![Enhancement][badge-enhancement] `lrp!` rule calls require extra argument `layer`, avoiding copies of unmodified layers ([#119])
- ![Enhancement][badge-enhancement] Performance fixes for LRP rules, reducing number of generated pullback functions ([#106], [#108])
- ![Enhancement][badge-enhancement] Simplify LRP analyzer ([#112], [#119])
- ![Enhancement][badge-enhancement] Simplify LRP model checks ([#110], [#119])
- ![Enhancement][badge-enhancement] Improve type stability of LRP rules

Documentation:
- ![Documentation][badge-docs] Update documentation, adding pages on model preparation, composites, custom LRP rules, developer documentation
  and a separate API reference for LRP analyzers ([#137], [#105])

Package maintenance:
- ![Maintenance][badge-maintenance] Refactor LRP rule tests ([#103], [#127])
- ![Maintenance][badge-maintenance] Fix LRP benchmarks ([#104], [#127])

### General changes
Breaking changes:
- ![BREAKING][badge-breaking] Rename `Explanation` field `attribution` to `val` ([#136])

Documentation:
- ![Documentation][badge-docs] Update documentation, adding pages on heatmapping and input augmentations ([#137], [#105])

Package maintenance:
- ![Maintenance][badge-maintenance] Compatibility with Flux.jl `v0.14` ([#116])
- ![Maintenance][badge-maintenance] Drop dependency on LinearAlgebra.jl and PrettyTables.jl ([#119])
- ![Maintenance][badge-maintenance] Add Aqua.jl tests ([#125])

## Version `v0.5.7`
- ![Bugfix][badge-bugfix] Fix `WSquareRule` dispatch on Dense layers
- ![Maintenance][badge-maintenance] Fix `Vararg` deprecation warnings from composites

## Version `v0.5.6`
- ![Bugfix][badge-bugfix] Drop Flux `v0.12` due to compatibility issues in `preactivation` ([#99])

## Version `v0.5.5`
- ![Bugfix][badge-bugfix] Ignore bias in `WSquareRule`
- ![Enhancement][badge-enhancement] Faster `FlatRule` on Dense layers ([#96])
- ![Enhancement][badge-enhancement] Faster `WSquareRule` on Dense layers ([#98])
- ![Maintenance][badge-maintenance] Update rule tests and references

## Version `v0.5.4`
This release brings bugfixes and usability features:
- ![Feature][badge-feature] Add pretty printing of LRP analyzers, summarizing how layers and rules are matched up ([#89])
- ![Feature][badge-feature] Add LRP support for `ConvTranspose` and `CrossCor` layers
- ![Documentation][badge-docs] Add equations of LRP rules to docstrings

Bugfixes:
- ![Bugfix][badge-bugfix] Fix bug affecting `AlphaBetaRule`, `ZPlusRule` and `ZBoxRule`, where mutating the layer modified Zygote pullbacks ([#92])
- ![Bugfix][badge-bugfix] Fix bug in `FlatRule` bias ([#92])
- ![Bugfix][badge-bugfix] Fix input modification for `FlatRule` and `WSquareRule` ([#93])


## Version `v0.5.3`
Big feature release that adds LRP composites and presets:
- ![Feature][badge-feature] Add LRP `Composite` and composite primitives ([#84]) 
- ![Feature][badge-feature] Add LRP composite presets ([#87])
- ![Feature][badge-feature] Add LRP `ZPlusRule` ([#88])
- ![Enhancement][badge-enhancement] Export union-types of Flux layers for easy definition of LRP composites
- ![Documentation][badge-docs] Improvements to docstrings and documentation
- ![Maintenance][badge-maintenance] Add `test/Project.toml` with compat entries for test dependencies ([#87])

## Version `v0.5.2`
This release temporarily adds ImageNet pre-processing utilities. This enables users users to apply XAI methods on pretrained vision models from Metalhead.jl. *Note that this functionality will be deprecated once matching functionality is in either Metalhead.jl or MLDatasets.jl.*
- ![Feature][badge-feature] Add ImageNet preprocessing utility `preprocess_imagenet` ([#80])
- ![Enhancement][badge-enhancement] Change default `heatmap` color scheme to `seismic`
- ![Enhancement][badge-enhancement] Updated README with the JuliaCon 2022 talk and examples on VGG16

## Version `v0.5.1`
Small bugfix release addressing a bug in `v0.5.0`. 
Version of ExplainableAI.jl shown in the JuliaCon 2022 talk.
- ![Bugfix][badge-bugfix] Fix bug in `FlatRule` ([#77])

## Version `v0.5.0`
Breaking release that refactors the internals of `LRP` analyzers and adds several rules.

List of breaking changes:
- ![BREAKING][badge-breaking]![Enhancement][badge-enhancement] Introduce compatibility checks for LRP rule & layer combinations using `check_compat(rule, layer)` ([#75])
- ![BREAKING][badge-breaking] Applying `GammaRule` and `ZBoxRule` on a layer without weights and biases will now throw an error ([#75])
- ![BREAKING][badge-breaking] In-place updating `modify_layer!(rule, layer)` replaces `modify_layer(rule, layer)` ([#73])
- ![BREAKING][badge-breaking] In-place updating `modify_param!(rule, param)` replaces `modify_params(rule, W, b)` ([#73])
- ![BREAKING][badge-breaking] Removed named LRP constructors `LRPZero`, `LRPEpsilon`, `LRPGamma` ([#75])

Added new LRP rules:
- ![Feature][badge-feature] Add `PassRule` ([#76])
- ![Feature][badge-feature] Add `AlphaBetaRule` ([#78])
- ![Feature][badge-feature] Add `FlatRule` ([a6e2c59][flat-wsquare-commit])
- ![Feature][badge-feature] Add `WSquareRule` ([a6e2c59][flat-wsquare-commit])

Bug fixes:
- ![Bugfix][badge-bugfix] Fix bug in `ZBoxRule` ([#77])
- ![Bugfix][badge-bugfix] Fix broadcasting for Julia 1.6 ([#74])
- ![Bugfix][badge-bugfix] Support `MLUtils.flatten`

Performance improvements:
- ![Enhancement][badge-enhancement] Replace LRP gradient computation with VJP using `Zygote.pullback` ([#72])
- ![Enhancement][badge-enhancement] Faster `GammaRule`

## Version `v0.4.0`
Changes:
- ![BREAKING][badge-breaking] Update heatmapping normalizer, using ColorScheme's `get`. Breaking due to renaming `normalize` to ColorScheme's `rangescale`. ([#57])
- ![BREAKING][badge-breaking] Rename `InputAugmentation` to `NoiseAugmentation`. ([#65])
- ![BREAKING][badge-breaking] `GammaRule` and `EpsilonRule` now use default arguments instead of keyword arguments, removing the need for users to type unicode symbols. ([#70]) 
- ![BREAKING][badge-breaking]![Bugfix][badge-bugfix] `ZBoxRule` now requires parameters `low` and `high` instead of computing them from the input. ([#69]) 
- ![Feature][badge-feature] Add `IntegratedGradients` analyzer. ([#65])
- ![Feature][badge-feature] Add `InterpolationAugmentation` wrapper. ([#65])
- ![Feature][badge-feature] Allow any type of `Sampleable` in `NoiseAugmentation`. ([#65])

Performance improvements:
- ![Enhancement][badge-enhancement] Remove use of `mapreduce`. ([#58])
- ![Enhancement][badge-enhancement] Load LoopVectorization.jl in tests and benchmarks to speed up Tullio on CPU. ([#66])
- ![Enhancement][badge-enhancement] Type stability fixes for `GammaRule`. ([#70])

<!--
# Badges
![BREAKING][badge-breaking]
![Deprecation][badge-deprecation]
![Feature][badge-feature]
![Enhancement][badge-enhancement]
![Bugfix][badge-bugfix]
![Experimental][badge-experimental]
![Maintenance][badge-maintenance]
![Documentation][badge-docs]
-->

[VisionHeatmaps]: https://julia-xai.github.io/XAIDocs/VisionHeatmaps/stable/
[TextHeatmaps]: https://julia-xai.github.io/XAIDocs/TextHeatmaps/stable/

[#177]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/177
[#174]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/174
[#167]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/167
[#166]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/166
[#162]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/162
[#159]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/159
[#157]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/157
[#155]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/155
[#154]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/154
[#150]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/150
[#149]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/149
[#148]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/148
[#147]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/147
[#146]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/146
[#145]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/145
[#144]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/144
[#142]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/142
[#141]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/141
[#140]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/140
[#138]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/138
[#137]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/137
[#136]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/136
[#135]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/135
[#134]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/134
[#131]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/131
[#130]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/130
[#129]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/129
[#128]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/128
[#127]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/127
[#126]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/126
[#125]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/125
[#119]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/119
[#116]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/116
[#112]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/112
[#110]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/110
[#109]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/109
[#108]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/108
[#107]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/107
[#106]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/106
[#105]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/105
[#104]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/104
[#103]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/103
[#102]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/102
[#99]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/99
[#98]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/98
[#96]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/96
[#93]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/93
[#92]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/92
[#89]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/89
[#88]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/88
[#87]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/87
[#84]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/84
[#80]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/80
[#78]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/78
[#77]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/77
[#76]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/76
[#75]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/75
[#74]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/74
[#73]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/73
[#72]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/72
[#70]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/70
[#69]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/69
[#67]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/67
[#66]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/66
[#65]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/65
[#58]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/58
[#57]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/57
[#26]: https://github.com/Julia-XAI/ExplainableAI.jl/pull/26

[flat-wsquare-commit]: https://github.com/Julia-XAI/ExplainableAI.jl/commit/a6e2c59094fe4f1d4b744123de79407ccbd4b972


[badge-breaking]: https://img.shields.io/badge/BREAKING-red.svg
[badge-deprecation]: https://img.shields.io/badge/deprecation-orange.svg
[badge-feature]: https://img.shields.io/badge/feature-green.svg
[badge-enhancement]: https://img.shields.io/badge/enhancement-blue.svg
[badge-bugfix]: https://img.shields.io/badge/bugfix-purple.svg
[badge-security]: https://img.shields.io/badge/security-black.svg
[badge-experimental]: https://img.shields.io/badge/experimental-lightgrey.svg
[badge-maintenance]: https://img.shields.io/badge/maintenance-gray.svg
[badge-docs]: https://img.shields.io/badge/docs-orange.svg
