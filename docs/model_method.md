# Waveform-Line Model Method

## Why plain binary segmentation

The current task target is already a single binary image:

```text
waveform PNG -> continuous vehicle-line mask
```

The label does not encode per-instance identity. A single-class segmentation
model is therefore the shortest path between supervision and prediction. A
query-instance model would introduce matching and per-instance ambiguity without
adding information that exists in the current labels.

## Network choice

The first model is a lightweight U-Net:

- input: one grayscale channel;
- output: one logit map;
- skip connections preserve fine line geometry;
- the depth is limited so CPU smoke tests stay practical.

All training inputs are resized to `512x512`. This reduces memory pressure and
keeps the pipeline stable before any patch-based or multi-scale training is
introduced.

## Loss

The positive class occupies very few pixels relative to the background. Using
plain BCE would bias the model toward predicting empty images. The first model
therefore combines:

- `BCEWithLogits` with a positive-class weight;
- soft Dice loss for overlap stability on thin structures.

## Postprocess and skeleton

Prediction first produces a binary mask through:

```text
sigmoid(logits) > threshold
```

The skeleton output is then computed from that mask by iterative thinning. The
current implementation uses a Zhang-Suen style binary thinning routine in pure
`numpy`, which avoids adding extra dependencies while still giving a usable
centerline image.

The first version returns:

- predicted mask PNG;
- predicted skeleton PNG;
- overlay preview PNG;
- a `summary.json` file with batch statistics.

