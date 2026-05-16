# Waveform-Line Model, Loss, Training Method, and Physical/Mathematical Principles

## 1. Task Definition

The current model solves a single binary segmentation task:

```text
waveform PNG -> vehicle-line binary mask
```

The supervision target is not an instance mask and does not contain vehicle IDs.
Each pixel only answers one question:

```text
Is this pixel on a visible vehicle trajectory centerline?
```

So the current implementation uses **single-class semantic segmentation**
instead of detection, instance segmentation, or track-query matching.

## 2. What Model Is Currently Used

The code in [model/network.py](/Volumes/SanDisk2T4/MyProjects/BaFang/KF/waveform_line_task/model/network.py)
defines the current model as `WaveformLineUNet`, a lightweight U-Net for
single-channel input and single-channel output.

Default configuration from `UNetConfig` and `train_model.py`:

| Item | Current default |
| --- | --- |
| Input channels | 1 |
| Output channels | 1 |
| Base channels | 32 |
| Depth | 4 downsampling stages + 4 upsampling stages |
| Dropout | 0.0 |
| Input image size | 512 x 512 |
| Output meaning | one logit per pixel |

In other words, the model predicts a dense score map

```math
z \in \mathbb{R}^{H \times W}
```

where `z(y, x)` is the pre-sigmoid confidence that pixel `(x, y)` belongs to a
vehicle line.

## 3. Why U-Net Fits This Task

This task is dominated by thin, connected, low-area structures:

- the target occupies only a very small fraction of all pixels;
- line continuity matters more than region area;
- local fine geometry matters, but global context also matters because the
  waveform texture around a line helps distinguish true trajectories from noise.

U-Net is a good fit because:

1. The encoder increases receptive field and captures larger waveform context.
2. The decoder reconstructs dense pixel predictions.
3. Skip connections preserve high-resolution line geometry from shallow layers.
4. The architecture is simple, stable, and practical on CPU and CUDA.

## 4. Network Structure

The architecture is:

```text
Input
  -> DoubleConv(stem)
  -> DownBlock 1
  -> DownBlock 2
  -> DownBlock 3
  -> DownBlock 4
  -> UpBlock 1 + skip
  -> UpBlock 2 + skip
  -> UpBlock 3 + skip
  -> UpBlock 4 + skip
  -> 1x1 Conv head
  -> Logits
```

### 4.1 DoubleConv block

Each `DoubleConv` contains:

1. `Conv2d(3x3, padding=1, bias=False)`
2. `GroupNorm`
3. `GELU`
4. `Conv2d(3x3, padding=1, bias=False)`
5. `GroupNorm`
6. `GELU`
7. optional `Dropout2d`

This is slightly more stable than plain conv + ReLU for small-batch training.
`GroupNorm` is especially useful when batch size is not large enough for
reliable batch-statistics estimation.

### 4.2 Downsampling path

Each `DownBlock` uses:

```text
MaxPool2d(2x2) -> DoubleConv
```

This halves spatial resolution and expands channels, allowing the model to
aggregate larger-scale waveform structure.

### 4.3 Upsampling path

Each `UpBlock` uses:

```text
ConvTranspose2d(2x2, stride=2) -> concatenate skip feature -> DoubleConv
```

The transposed convolution restores spatial resolution, and the skip tensor
injects fine shallow-layer detail back into the decoder.

### 4.4 Output layer

The final head is a `1x1` convolution producing one channel:

```math
z = f_\theta(x)
```

where:

- `x` is the grayscale waveform image;
- `f_\theta` is the U-Net with parameters `\theta`;
- `z` is the per-pixel logit map.

The probability map is:

```math
p = \sigma(z) = \frac{1}{1 + e^{-z}}
```

## 5. Data Representation the Model Learns From

The dataset loader in
[model/dataset.py](/Volumes/SanDisk2T4/MyProjects/BaFang/KF/waveform_line_task/model/dataset.py)
reads:

- grayscale waveform image PNG;
- grayscale label PNG.

Then it converts them to tensors in `[0, 1]` and resizes both to the training
resolution.

Important implementation details:

- input image resize uses bilinear interpolation;
- label resize uses nearest-neighbor interpolation;
- labels are thresholded back to binary after resize.

This is correct because labels represent class membership, not continuous
intensity. Nearest-neighbor resize avoids creating fake soft boundaries caused
by interpolation.

## 6. Current Loss Function

The file
[model/losses.py](/Volumes/SanDisk2T4/MyProjects/BaFang/KF/waveform_line_task/model/losses.py)
defines the current segmentation loss as:

```math
\mathcal{L}
= \lambda_{\text{BCE}} \mathcal{L}_{\text{BCE}}
+ \lambda_{\text{Dice}} \mathcal{L}_{\text{Dice}}
```

with current defaults:

| Term | Default |
| --- | --- |
| `bce_weight` | 1.0 |
| `dice_weight` | 1.0 |
| `positive_class_weight` | 6.0 |

So the actual default loss is:

```math
\mathcal{L} = \mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Dice}}
```

with positive samples reweighted inside BCE.

### 6.1 BCEWithLogits term

The BCE term uses `binary_cross_entropy_with_logits`, which means the sigmoid is
fused numerically with the binary cross-entropy.

For a single pixel with target `y \in \{0,1\}` and logit `z`, the weighted BCE
conceptually corresponds to:

```math
\mathcal{L}_{\text{BCE}}(z, y)
= - w_+ y \log \sigma(z) - (1-y)\log(1-\sigma(z))
```

where:

- `\sigma(z)` is the sigmoid probability;
- `w_+ = 6.0` is the positive-class weight by default.

#### Why the positive-class weight is needed

Vehicle-line pixels are sparse. If the model minimizes ordinary BCE on a highly
imbalanced image, it can reduce loss simply by predicting background almost
everywhere. Positive reweighting increases the penalty for missing line pixels,
so the optimizer cannot cheaply collapse to empty masks.

### 6.2 Soft Dice term

The Dice loss is implemented on sigmoid probabilities:

```math
p = \sigma(z)
```

For one sample:

```math
\text{Dice}(p, y)
= \frac{2\sum_i p_i y_i + \varepsilon}
       {\sum_i p_i + \sum_i y_i + \varepsilon}
```

and the loss is:

```math
\mathcal{L}_{\text{Dice}} = 1 - \text{Dice}(p, y)
```

#### Why Dice is added

BCE is a pixelwise classification objective. It does not directly optimize
region overlap. Dice compensates for that by emphasizing agreement on the sparse
foreground set. This is particularly useful for thin structures where a small
number of missed pixels can break a line visually and topologically.

### 6.3 Why BCE + Dice together

The combination balances two things:

- BCE provides stable local per-pixel gradients.
- Dice emphasizes global foreground overlap and resists extreme class imbalance.

For line segmentation, this combination is usually more reliable than either
term alone.

## 7. Current Training Method

The training entry point is
[train_model.py](/Volumes/SanDisk2T4/MyProjects/BaFang/KF/waveform_line_task/train_model.py).

### 7.1 Default training hyperparameters

| Item | Current default |
| --- | --- |
| Epochs | 60 |
| Batch size | 12 |
| Learning rate | `2e-4` |
| Weight decay | `1e-4` |
| Validation fraction | 0.1 |
| Threshold for metrics | 0.5 |
| Random seed | 42 |
| Optimizer | AdamW |
| Scheduler | none |

One important point: **the current training code does not use any learning-rate
scheduler**. The learning rate stays fixed unless the user changes it manually.

### 7.2 Dataset split method

The split is performed by `split_records(records, val_fraction)`:

- it preserves the manifest order;
- it reserves the tail portion for validation.

So this is not a random shuffled split inside the function itself. Reproducible
ordering is therefore part of the current method.

### 7.3 One training iteration

For each batch:

1. Load waveform image tensor and binary target mask.
2. Move tensors to the selected device.
3. Run forward pass to get logits.
4. Compute `BCE + Dice` loss.
5. Compute monitoring metrics.
6. Backpropagate.
7. Update parameters with AdamW.

In formula form:

```math
z = f_\theta(x), \quad
\mathcal{L} = \mathcal{L}(z, y), \quad
\theta \leftarrow \theta - \eta \cdot \text{AdamW}(\nabla_\theta \mathcal{L})
```

### 7.4 Optimizer: AdamW

The optimizer is:

```python
torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
```

AdamW combines:

- first-moment estimation of gradients;
- second-moment estimation of gradient magnitudes;
- decoupled weight decay regularization.

Mathematically, Adam-style optimization maintains moving averages:

```math
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
```

```math
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
```

and updates parameters using normalized gradients. AdamW separates weight decay
from the gradient term, which generally behaves better than naive L2 mixing in
adaptive optimizers.

### 7.5 Mixed precision and hardware path

Current runtime logic:

- `--device auto` prefers CUDA, then MPS, then CPU;
- CUDA training enables `cuDNN benchmark`;
- CUDA enables TF32 matmul and TF32 cuDNN;
- CUDA uses channels-last memory format;
- CUDA uses AMP by default unless `--no-amp` is set;
- CPU fallback remains available.

AMP uses `autocast` and `GradScaler` on CUDA. This reduces memory bandwidth and
often increases throughput while preserving stability through dynamic gradient
scaling.

### 7.6 What is not currently used

The current script does **not** use:

- learning rate warmup;
- cosine schedule or step schedule;
- EMA model averaging;
- gradient clipping;
- deep supervision;
- focal loss;
- data augmentation in the current dataset loader.

That matters because the document should reflect the real code path, not a
generic training recipe.

## 8. Validation, Checkpointing, and Logging

### 8.1 Validation strategy

If a validation split exists, validation runs every epoch.

However, expensive skeleton metrics are only enabled every
`val_skeleton_every` epochs, default `10`.

This is a deliberate throughput tradeoff: skeletonization is CPU-heavy and can
reduce GPU utilization if computed too often.

### 8.2 Checkpoint policy

The script writes:

- `checkpoint_last.pt`
- `checkpoint_best.pt`
- `train_config.json`
- `train_history.jsonl`

An implementation detail worth noting:

- periodic checkpoint saving is tied to `val_skeleton_every`;
- `checkpoint_best.pt` is only updated on those checkpoint epochs;
- after the training loop ends, `checkpoint_last.pt` is saved one more time.

So "best" currently means best among the scheduled checkpoint epochs, not
necessarily best among every single epoch.

### 8.3 Preview export

When skeleton validation is enabled for an epoch, the script also exports one
validation preview image showing:

- raw input;
- target overlay;
- prediction overlay.

This is used for qualitative inspection of failure modes such as broken lines,
false positives, or line thickening.

## 9. Evaluation Metrics

The file
[model/metrics.py](/Volumes/SanDisk2T4/MyProjects/BaFang/KF/waveform_line_task/model/metrics.py)
computes two families of metrics.

### 9.1 Pixel-level metrics

After thresholding:

```math
\hat{y} = \mathbf{1}[\sigma(z) \ge 0.5]
```

the script computes:

- precision
- recall
- F1
- IoU
- Dice

using:

```math
\text{Precision} = \frac{TP}{TP + FP}
```

```math
\text{Recall} = \frac{TP}{TP + FN}
```

```math
F1 = \frac{2PR}{P + R}
```

```math
\text{IoU} = \frac{TP}{TP + FP + FN}
```

```math
\text{Dice} = \frac{2TP}{2TP + FP + FN}
```

These evaluate segmentation accuracy directly in image space.

### 9.2 Skeleton metrics

For line tasks, thick-mask overlap is not the full story. Two masks can have
similar area overlap but different centerline quality. So the code also
skeletonizes both prediction and target, then computes:

- skeleton precision
- skeleton recall
- skeleton F1

These are closer to the real objective of extracting correct trajectory
centerlines.

## 10. Postprocessing and Skeleton Extraction

The file
[model/postprocess.py](/Volumes/SanDisk2T4/MyProjects/BaFang/KF/waveform_line_task/model/postprocess.py)
implements two main steps.

### 10.1 Thresholding

The raw network output is a logit map. It is converted to a binary mask by:

```math
p = \sigma(z), \quad \hat{y} = \mathbf{1}[p \ge \tau]
```

with current default threshold:

```math
\tau = 0.5
```

### 10.2 Zhang-Suen skeletonization

The binary mask is then thinned to a 1-pixel-like centerline using a
Zhang-Suen style iterative thinning algorithm.

Its core idea is:

1. Scan foreground pixels.
2. Remove boundary pixels that do not break connectivity.
3. Alternate two deletion substeps.
4. Repeat until no more pixels can be safely removed.

This is a topological thinning method. It tries to preserve connected structure
while reducing thick regions to a medial line approximation.

That makes it useful here because the final physical object of interest is not a
filled area, but a trajectory centerline.

## 11. Physical Principle Behind the Task

The model is image-based, but the labels come from a physical generative model
described in
[docs/method.md](/Volumes/SanDisk2T4/MyProjects/BaFang/KF/waveform_line_task/docs/method.md).

### 11.1 DAS-like channel-time representation

The underlying synthetic signal is a channel-time field:

```math
X[c, t]
```

where:

- `c` is channel index, corresponding to position along the fiber;
- `t` is time.

Each vehicle contributes a localized waveform pulse around the channel crossing
time `\tau_c`.

### 11.2 Pulse model

A simplified pulse contribution is:

```math
A \exp\left(-\frac{(t - \tau_c)^2}{2\sigma^2}\right)
```

where:

- `A` is amplitude;
- `\sigma` controls temporal spread;
- `\tau_c` is the crossing time at channel `c`.

Multiple vehicles and noise sources are superposed, so the observed waveform is
the sum of structured signal and disturbances.

### 11.3 Geometry of a moving vehicle

If channel spacing is `dx_m` and vehicle speed is `v` in km/h, then adjacent
channel crossing times differ by:

```math
\Delta t = \frac{dx_m}{v / 3.6}
```

because `v / 3.6` converts km/h to m/s.

This equation is the direct physical reason trajectory lines appear slanted in
the channel-time image:

- higher speed -> smaller `\Delta t` -> flatter line;
- lower speed -> larger `\Delta t` -> steeper line;
- reverse direction changes the sign of time progression across channels.

So the label is not arbitrary annotation. It is the image-space projection of a
physical propagation trajectory.

### 11.4 Why line extraction is physically meaningful

The centerline encodes:

- direction of travel;
- relative speed through slope;
- continuity through visible channels;
- interaction patterns such as crossing and near-parallel motion.

Therefore, extracting the line mask is equivalent to estimating the geometric
trace of moving vehicles in channel-time coordinates.

## 12. Mathematical Interpretation of the Whole Pipeline

The full pipeline can be summarized as:

1. A physical generative process produces waveform image `x` and binary target
   mask `y`.
2. The neural network learns a parametric mapping:

   ```math
   f_\theta : x \mapsto z
   ```

3. The sigmoid converts logits to probabilities:

   ```math
   p = \sigma(f_\theta(x))
   ```

4. Training solves:

   ```math
   \theta^\star = \arg\min_\theta \mathbb{E}_{(x,y)\sim \mathcal{D}}
   \left[
   \lambda_{\text{BCE}}\mathcal{L}_{\text{BCE}}(f_\theta(x), y)
   +
   \lambda_{\text{Dice}}\mathcal{L}_{\text{Dice}}(f_\theta(x), y)
   \right]
   ```

5. Inference thresholds `p` and optionally skeletonizes the result to recover a
   centerline estimate.

This is a supervised empirical risk minimization problem under strong class
imbalance and thin-structure topology constraints.

## 13. Practical Summary

The current codebase is using:

- **Model**: lightweight single-channel U-Net
- **Loss**: weighted BCEWithLogits + soft Dice
- **Optimizer**: AdamW
- **Scheduler**: none
- **Training resolution**: `512 x 512`
- **Threshold**: `0.5`
- **Best suited target**: sparse, thin vehicle trajectory centerlines
- **Postprocess**: sigmoid thresholding + Zhang-Suen skeleton thinning

The main physical idea is that a moving vehicle traces a structured line in the
channel-time plane. The main mathematical idea is to learn that line as a
sparse binary segmentation problem with imbalance-aware overlap-constrained
optimization.
