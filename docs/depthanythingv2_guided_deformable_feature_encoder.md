# DepthAnythingV2-Guided Deformable Feature Encoder

## Goal

Modify the DEFOM feature encoder so that `DepthAnythingV2` features actively guide low-level stereo feature extraction, instead of only being fused by additive injection after the image stem. The proposed direction is to replace the first `7x7` image convolution in `BasicEncoder` with a deformable convolution whose sampling pattern is predicted from `DepthAnythingV2` outputs.

This note focuses on the feature branch:

- current path: `DefomEncoder -> dfeat1/dfeat2 -> BasicEncoder`
- target path: `DefomEncoder -> guide head -> deformable stem in BasicEncoder`


## Current Baseline

In the current implementation:

- `DefomEncoder` returns `d_features`, `dfeat1`, `dfeat2`, and initial disparity
- `BasicEncoder` applies:
  - `conv1: 7x7` on RGB
  - residual blocks
  - `x = x + self.convd(dfeats)`

So `DepthAnythingV2` features influence `fnet`, but only after the first image convolution and only through a residual additive path. They do not change where the image branch samples from.


## Core Idea

Use `DepthAnythingV2` features to predict deformable convolution parameters for the first stem layer of `BasicEncoder`, so that the feature encoder can adapt its receptive field to scene geometry before correlation features are built.

Instead of:

```text
RGB --7x7 conv--> early feature
```

use:

```text
DepthAnythingV2 feature --guide head--> offsets / modulation
RGB --deformable 7x7 conv--> geometry-aware early feature
```


## Why This Is Reasonable

The first stem layer has the largest effect on downstream texture extraction because:

- it sees raw image gradients
- it sets the spatial support used by later residual layers
- stereo matching is sensitive to edges, slanted surfaces, and low-texture regions

If the sampling grid can shift based on monocular geometry priors, the encoder may:

- align sampling to slanted disparity structures
- widen support in textureless regions
- avoid mixing across depth discontinuities
- build correlation features that are more geometry-aware from the start


## Recommended Design

### Option A: Offset-Only Deformable Stem

Replace `BasicEncoder.conv1` with deformable conv and predict only offsets from `dfeat`.

Structure:

```text
dfeat -> 3x3 conv -> ReLU -> 3x3 conv -> offsets
RGB -> DeformConv2d(7x7, offsets) -> norm -> relu
```

For a `7x7` deformable convolution:

- number of sampling locations = `49`
- offset channels = `2 * 49 = 98`

If using DCNv2-style modulation:

- mask channels = `49`

So parameter counts are:

- offset-only: `98`
- offset + modulation mask: `147`

This is the cleanest first version. It is easier to stabilize and easier to compare against baseline.

Recommendation: start here.


### Option B: Offset + Modulation Stem

Predict both offsets and modulation mask from `dfeat`.

Structure:

```text
dfeat -> guide head -> [offset_x, offset_y, mask]
RGB -> ModulatedDeformConv2d(7x7, offsets, mask)
```

Potential gain:

- mask can suppress unreliable shifted samples
- more expressive than offset-only

Potential cost:

- more sensitive to training instability
- may collapse to trivial masks early in training

Recommendation: use only after offset-only version is working.


### Option C: Residual Deformable Stem

Use both a standard and a deformable branch:

```text
stem_std = Conv7x7(RGB)
stem_def = DeformConv7x7(RGB, guide(dfeat))
alpha = sigmoid(head(dfeat))
stem = stem_std + alpha * stem_def
```

Advantages:

- preserves baseline behavior
- easier optimization
- safer if `DepthAnythingV2` guidance is noisy

This is my preferred production structure if you want lower risk.


## Strong Recommended Architecture

If the target is "good chance of working without excessive instability", use this:

```text
DepthAnythingV2 dfeat
  -> guide_reduce: 3x3 conv to 64
  -> guide_refine: residual block
  -> offset_head: 3x3 conv -> 98 channels
  -> mask_head: 3x3 conv -> 49 channels
  -> alpha_head: 3x3 conv -> 1 channel

RGB
  -> std_stem: 7x7 conv, out 64
  -> def_stem: modulated deformable 7x7 conv, out 64

stem = std_stem + sigmoid(alpha) * def_stem
```

Then continue with existing normalization and residual tower.

This keeps three good properties:

- baseline path still exists
- geometry-guided sampling is available
- the model can learn how much to trust the guided branch


## What Should Be Used As Guide Feature

For `BasicEncoder`, the direct guide should be `dfeat1` and `dfeat2`, not `d_features`.

Reason:

- `BasicEncoder` processes left and right image branches separately
- `dfeat1` / `dfeat2` already correspond to left/right image-specific features
- `d_features` is used by `cnet` as a multi-scale context feature bundle, which is a different role

So the intended mapping is:

- left image feature stem guided by `dfeat1`
- right image feature stem guided by `dfeat2`


## Resolution Alignment

The guide feature used for deformable parameters must match the spatial resolution expected by the first stem layer output.

Current `BasicEncoder.conv1` behavior depends on `downsample`:

- if `downsample > 2`, stride is `2`
- otherwise stride is `1`

That means your guide head should predict offsets at the output resolution of the stem, not necessarily at full RGB resolution.

Recommended practice:

1. run the guide head on `dfeat`
2. interpolate guide features if needed
3. predict offsets/masks at the exact spatial size of `conv1` output

Do not predict full-resolution offsets and then downsample them blindly.


## How To Inject Depth Prior

There are three sensible ways to use `DepthAnythingV2` output for deformable parameters.

### 1. Feature-Only Guidance

Use only `dfeat`.

Pros:

- simplest
- avoids hard dependence on monocular disparity scale

Cons:

- less explicit geometry signal

This should be your default first experiment.


### 2. Feature + Initial Disparity Guidance

Concatenate `dfeat` with `disp_init`.

```text
guide_input = concat(dfeat, disp_init)
```

Pros:

- explicit depth structure can influence offsets
- useful for slanted planes and depth discontinuities

Cons:

- may overfit to monocular disparity bias

This is a strong second experiment.


### 3. Feature + Edge Guidance

Derive a discontinuity prior:

- image gradient
- depth gradient
- confidence-like edge magnitude from `disp_init`

Then use it to gate offset magnitude.

Example:

```text
offset = raw_offset * edge_gate
```

This is useful if you want to prevent large, noisy offset jumps in flat regions.


## Offset Constraints

This part matters a lot for training stability.

Do not let the offset head produce unconstrained large values from the start.

Recommended constraint:

```text
offset = offset_range * tanh(raw_offset)
```

Suggested initial range:

- `offset_range = 2.0` for `7x7`
- try `1.0` first if optimization is unstable

For modulation mask:

```text
mask = sigmoid(raw_mask)
```

For residual blending weight:

```text
alpha = sigmoid(raw_alpha)
```


## Initialization Strategy

For a stable first run:

- initialize deformable offset head weights and bias to zero
- initialize modulation mask bias so mask starts near `1`
- initialize alpha bias so deformable branch starts weak

Practical effect:

- initial behavior stays close to standard convolution
- guided sampling is learned gradually

This is important if you resume from an existing DEFOM checkpoint.


## Where To Modify The Code

Primary files:

- `core/extractor.py`
- `core/defom_stereo.py`

Main insertion points:

1. In `BasicEncoder`
   - replace `self.conv1`
   - add guide heads for offsets / mask / alpha
   - update `forward(self, x, dfeats)`

2. In `DefomEncoder`
   - keep returning `dfeat1`, `dfeat2`
   - optionally also return an extra guide tensor if needed

3. In `DEFOMStereo.forward`
   - no large logic change is required if `BasicEncoder` still accepts `([image1, image2], [dfeat1, dfeat2])`

If you later want the same idea in `cnet`, then modify `MultiBasicEncoder` separately. Do not couple both changes in the first experiment.


## Suggested Implementation Order

### Stage 1

Implement offset-only deformable stem in `BasicEncoder`.

Keep everything else unchanged:

- `cnet` unchanged
- update blocks unchanged
- output unchanged

This isolates the effect to `fnet`.


### Stage 2

Add residual blending:

```text
stem = std + alpha * def
```

Compare:

- standard stem
- pure deformable stem
- residual deformable stem


### Stage 3

Test guide input variants:

- `dfeat`
- `concat(dfeat, disp_init)`


### Stage 4

Only if useful, extend the same guided deformable stem idea to `MultiBasicEncoder` for `cnet`.


## Recommended Ablations

At minimum, compare these:

1. Baseline DEFOM
2. Offset-only deformable `fnet` stem
3. Offset+mask deformable `fnet` stem
4. Residual deformable `fnet` stem
5. Residual deformable `fnet` stem with `dfeat + disp_init` guide

Measure:

- SceneFlow validation
- Middlebury H/F
- KITTI if cross-domain robustness matters
- runtime and memory overhead


## Main Risks

### Risk 1: Monocular prior overwhelms stereo evidence

If guide features dominate too early, `fnet` may become biased toward monocular guesses instead of stereo consistency.

Mitigation:

- keep standard stem branch
- use weak initial alpha
- start with offset-only or residual blend


### Risk 2: Offset noise hurts matching precision

Stereo depends on fine spatial alignment. Poor offsets can blur correspondence cues.

Mitigation:

- constrain offsets with `tanh`
- initialize offsets to zero
- optionally regularize offset magnitude


### Risk 3: Left-right asymmetry becomes harmful

If left and right guide branches diverge too aggressively, correlation quality may degrade.

Mitigation:

- share the guide-head architecture across views
- keep parameterization symmetric
- inspect offset statistics for both branches


## Best First Version

If the goal is a practical first implementation, I recommend:

1. Modify only `BasicEncoder`
2. Use `dfeat1` / `dfeat2` as guide input
3. Use residual deformable stem
4. Predict offset + mask + alpha
5. Initialize offsets to zero and alpha small

That gives a good tradeoff between novelty, controllability, and compatibility with the current codebase.


## Minimal Pseudocode

```python
guide = guide_reduce(dfeat)
guide = guide_refine(guide)

offset = offset_range * torch.tanh(offset_head(guide))
mask = torch.sigmoid(mask_head(guide))
alpha = torch.sigmoid(alpha_head(guide))

stem_std = conv1_std(x)
stem_def = deform_conv1(x, offset, mask)
stem = stem_std + alpha * stem_def

stem = norm1(stem)
stem = relu1(stem)
```


## Recommendation Summary

Do not start by making both `feature encoder` and `context encoder` deformable. Start with the first `7x7` stem in `BasicEncoder`, guided by `dfeat1/dfeat2`, using a residual deformable branch with conservative initialization. That is the cleanest experiment with the highest signal-to-risk ratio.
