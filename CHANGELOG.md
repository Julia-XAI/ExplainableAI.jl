# ExplainableAI.jl
## Version `v0.5.5`
- ![Bugfix][badge-bugfix] Ignore bias in `WSquareRule`
- ![Enhancement][badge-enhancement] Faster `FlatRule` on Dense layers ([#96][pr-96])
- ![Enhancement][badge-enhancement] Faster `WSquareRule` on Dense layers ([#98][pr-98])
- ![Maintenance][badge-maintenance] Update rule tests and references

## Version `v0.5.4`
This release brings bugfixes and usability features:
- ![Feature][badge-feature] Add pretty printing of LRP analyzers, summarizing how layers and rules are matched up ([#89][pr-89])
- ![Feature][badge-feature] Add LRP support for `ConvTranspose` and `CrossCor` layers
- ![Documentation][badge-docs] Add equations of LRP rules to docstrings

Bugfixes:
- ![Bugfix][badge-bugfix] Fix bug affecting `AlphaBetaRule`, `ZPlusRule` and `ZBoxRule`, where mutating the layer modified Zygote pullbacks ([#92][pr-92])
- ![Bugfix][badge-bugfix] Fix bug in `FlatRule` bias ([#92][pr-92])
- ![Bugfix][badge-bugfix] Fix input modification for `FlatRule` and `WSquareRule` ([#93][pr-93])


## Version `v0.5.3`
Big feature release that adds LRP composites and presets:
- ![Feature][badge-feature] Add LRP `Composite` and composite primitives ([#84][pr-84]) 
- ![Feature][badge-feature] Add LRP composite presets ([#87][pr-87])
- ![Feature][badge-feature] Add LRP `ZPlusRule` ([#88][pr-88])
- ![Enhancement][badge-enhancement] Export union-types of Flux layers for easy definition of LRP composites
- ![Documentation][badge-docs] Improvements to docstrings and documentation
- ![Maintenance][badge-maintenance] Add `test/Project.toml` with compat entries for test dependencies ([#87][pr-87])

## Version `v0.5.2`
This release temporarily adds ImageNet pre-processing utilities. This enables users users to apply XAI methods on pretrained vision models from Metalhead.jl. *Note that this functionality will be deprecated once matching functionality is in either Metalhead.jl or MLDatasets.jl.*
- ![Feature][badge-feature] Add ImageNet preprocessing utility `preprocess_imagenet` ([#80][pr-80])
- ![Enhancement][badge-enhancement] Change default `heatmap` color scheme to `seismic`
- ![Enhancement][badge-enhancement] Updated README with the JuliaCon 2022 talk and examples on VGG16

## Version `v0.5.1`
Small bugfix release adressing a bug in `v0.5.0`. 
Version of ExplainableAI.jl shown in the JuliaCon 2022 talk.
- ![Bugfix][badge-bugfix] Fix bug in `FlatRule` ([#77][pr-77])

## Version `v0.5.0`
Breaking release that refactors the internals of `LRP` analyzers and adds several rules.

List of breaking changes:
- ![BREAKING][badge-breaking]![Enhancement][badge-enhancement] Introduce compatibility checks for LRP rule & layer combinations using `check_compat(rule, layer)` ([#75][pr-75])
- ![BREAKING][badge-breaking] Applying `GammaRule` and `ZBoxRule` on a layer without weights and biases will now throw an error ([#75][pr-75])
- ![BREAKING][badge-breaking] In-place updating `modify_layer!(rule, layer)` replaces `modify_layer(rule, layer)` ([#73][pr-73])
- ![BREAKING][badge-breaking] In-place updating `modify_param!(rule, param)` replaces `modify_params(rule, W, b)` ([#73][pr-73])
- ![BREAKING][badge-breaking] Removed named LRP constructors `LRPZero`, `LRPEpsilon`, `LRPGamma` ([#75][pr-75])

Added new LRP rules:
- ![Feature][badge-feature] Add `PassRule` ([#76][pr-76])
- ![Feature][badge-feature] Add `AlphaBetaRule` ([#78][pr-78])
- ![Feature][badge-feature] Add `FlatRule` ([a6e2c59][flat-wsquare-commit])
- ![Feature][badge-feature] Add `WSquareRule` ([a6e2c59][flat-wsquare-commit])

Bug fixes:
- ![Bugfix][badge-bugfix] Fix bug in `ZBoxRule` ([#77][pr-77])
- ![Bugfix][badge-bugfix] Fix broadcasting for Julia 1.6 ([#74][pr-74])
- ![Bugfix][badge-bugfix] Support `MLUtils.flatten`

Performance improvements:
- ![Enhancement][badge-enhancement] Replace LRP gradient computation with VJP using `Zygote.pullback` ([#72][pr-72])
- ![Enhancement][badge-enhancement] Faster `GammaRule`

## Version `v0.4.0`
Changes:
- ![BREAKING][badge-breaking] Update heatmapping normalizer, using ColorScheme's `get`. Breaking due to renaming `normalize` to ColorScheme's `rangescale`. ([#57][pr-57])
- ![BREAKING][badge-breaking] Rename `InputAugmentation` to `NoiseAugmentation`. ([#65][pr-65])
- ![BREAKING][badge-breaking] `GammaRule` and `EpsilonRule` now use default arguments instead of keyword arguments, removing the need for users to type unicode symbols. ([#70][pr-70]) 
- ![BREAKING][badge-breaking]![Bugfix][badge-bugfix] `ZBoxRule` now requires parameters `low` and `high` instead of computing them from the input. ([#69][pr-69]) 
- ![Feature][badge-feature] Add `IntegratedGradients` analyzer. ([#65][pr-65])
- ![Feature][badge-feature] Add `InterpolationAugmentation` wrapper. ([#65][pr-65])
- ![Feature][badge-feature] Allow any type of `Sampleable` in `NoiseAugmentation`. ([#65][pr-65])

Performance improvements:
- ![Enhancement][badge-enhancement] Remove use of `mapreduce`. ([#58][pr-58])
- ![Enhancement][badge-enhancement] Load LoopVectorization.jl in tests and benchmarks to speed up Tullio on CPU. ([#66][pr-66])
- ![Enhancement][badge-enhancement] Type stability fixes for `GammaRule`. ([#70][pr-70])

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

[pr-70]: https://github.com/adrhill/ExplainableAI.jl/pull/70
[pr-69]: https://github.com/adrhill/ExplainableAI.jl/pull/69
[pr-67]: https://github.com/adrhill/ExplainableAI.jl/pull/67
[pr-66]: https://github.com/adrhill/ExplainableAI.jl/pull/66
[pr-65]: https://github.com/adrhill/ExplainableAI.jl/pull/65
[pr-58]: https://github.com/adrhill/ExplainableAI.jl/pull/58
[pr-57]: https://github.com/adrhill/ExplainableAI.jl/pull/57
[pr-58]: https://github.com/adrhill/ExplainableAI.jl/pull/57
[pr-65]: https://github.com/adrhill/ExplainableAI.jl/pull/65
[pr-66]: https://github.com/adrhill/ExplainableAI.jl/pull/66
[pr-67]: https://github.com/adrhill/ExplainableAI.jl/pull/67
[pr-69]: https://github.com/adrhill/ExplainableAI.jl/pull/69
[pr-70]: https://github.com/adrhill/ExplainableAI.jl/pull/70
[pr-72]: https://github.com/adrhill/ExplainableAI.jl/pull/72
[pr-73]: https://github.com/adrhill/ExplainableAI.jl/pull/73
[pr-74]: https://github.com/adrhill/ExplainableAI.jl/pull/74
[pr-75]: https://github.com/adrhill/ExplainableAI.jl/pull/75
[pr-76]: https://github.com/adrhill/ExplainableAI.jl/pull/76
[pr-77]: https://github.com/adrhill/ExplainableAI.jl/pull/77
[pr-78]: https://github.com/adrhill/ExplainableAI.jl/pull/78
[pr-80]: https://github.com/adrhill/ExplainableAI.jl/pull/80
[pr-84]: https://github.com/adrhill/ExplainableAI.jl/pull/84
[pr-87]: https://github.com/adrhill/ExplainableAI.jl/pull/87
[pr-88]: https://github.com/adrhill/ExplainableAI.jl/pull/88
[pr-89]: https://github.com/adrhill/ExplainableAI.jl/pull/89
[pr-92]: https://github.com/adrhill/ExplainableAI.jl/pull/92
[pr-93]: https://github.com/adrhill/ExplainableAI.jl/pull/93
[pr-96]: https://github.com/adrhill/ExplainableAI.jl/pull/96
[pr-98]: https://github.com/adrhill/ExplainableAI.jl/pull/98

[flat-wsquare-commit]: https://github.com/adrhill/ExplainableAI.jl/commit/a6e2c59094fe4f1d4b744123de79407ccbd4b972


[badge-breaking]: https://img.shields.io/badge/BREAKING-red.svg
[badge-deprecation]: https://img.shields.io/badge/deprecation-orange.svg
[badge-feature]: https://img.shields.io/badge/feature-green.svg
[badge-enhancement]: https://img.shields.io/badge/enhancement-blue.svg
[badge-bugfix]: https://img.shields.io/badge/bugfix-purple.svg
[badge-security]: https://img.shields.io/badge/security-black.svg
[badge-experimental]: https://img.shields.io/badge/experimental-lightgrey.svg
[badge-maintenance]: https://img.shields.io/badge/maintenance-gray.svg
[badge-docs]: https://img.shields.io/badge/docs-orange.svg
