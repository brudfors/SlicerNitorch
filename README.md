# SliceNitorch: 3D Slicer Module for Image Registration

GPU-accelerated affine + nonlinear 3D image registration powered by [NITorch](https://github.com/balbasty/nitorch) in [3D Slicer](https://download.slicer.org/).

![SlicerNitorch](SlicerNitorch.png)

## Prerequisites

- 3D Slicer >= 5.x
- NVIDIA GPU (optional but recommended for faster registration)

### 1. Install the SlicerPyTorch extension

`torch_interpol` and `nitorch` require PyTorch, which Slicer does not bundle and which is managed by the **SlicerPyTorch** extension. Open the **Extensions Manager**, search for **PyTorch**, click **Install**, and restart Slicer when prompted.

### 2. Install nitorch (recommended: the installer script)

Clone or download this repository, then in the Python console (`View > Python Console`) run:

```python
import sys; sys.path.insert(0, "/path/to/SlicerNitorch/Scripts")
import install_nitorch
install_nitorch.run()
```

**To get the compiled backend, two things must already be installed on the system:** an NVIDIA **CUDA toolkit** (provides `nvcc`, found under `/usr/local/cuda-*`) and a **C++ compiler** (on Ubuntu: `sudo apt install build-essential`). With those present, `install_nitorch.run()` does everything else automatically; without them it installs the TS backend.

> ⏱️ **The compiled build is slow** — it compiles many CPU + CUDA kernels, so **20–30 minutes is not unusual**.

If a CUDA toolkit or C++ compiler isn't available, it installs the pure (TorchScript / "TS") backend instead and prints how to enable the compiled one. **Windows and macOS** currently use the TS backend (native compiled builds there are not yet supported). Use `install_nitorch.run(force="pure")` to skip compiling. **Restart Slicer** afterwards so the new nitorch is picked up.

### 3. Add the module path

1. Open 3D Slicer
2. Go to **Edit > Application Settings > Modules**
3. Under **Additional module paths**, click **Add** and browse to:
   ```
   /path/to/SlicerNitorch
   ```
4. Restart 3D Slicer

The module will appear under **Modules > Registration > NITorch Register**.

## Usage

The module has two tabs: **Registration** and **Validation**.

### Registration Tab

1. Load fixed and moving volumes/labels into Slicer
2. Open **NITorch Register** from the Modules menu
3. Select the fixed and moving volumes/labels
4. Choose a **Registration Mode** (see below; defaults to Automatic)
5. In Manual mode: check **Categorical** for label maps and pick a loss function (LCC, MSE, NMI, or Dice)
6. Select the computation device (CPU or CUDA; defaults to the first CUDA device if available)
7. Click **Run Registration**

The module creates a grid transform node. If **Apply output transform to moving** is checked, the transform is automatically applied to the moving volume.

#### Registration Mode

- **Automatic (NMI affine → LCC)** *(default)* — a two-pass pipeline for intensity images: first a pure-affine pass with the **NMI** loss (robust initial alignment), then a **combined affine + nonlinear (SVF)** pass with the **LCC** loss, warm-started from the NMI affine (which is then free to refine). Both passes use the same coarse-to-fine pyramid. It is intensity-only, so the **Categorical**, **Loss Function**, and **Affine Only** controls are disabled and the **Parameters** section is collapsed in this mode (its values are still used). Automatic mode produces **two** transforms — the intermediate affine (`NITorch_NNN_auto_affine_...`) and the final result (`NITorch_NNN_auto_nonlin_...`) — and selects the final one.
- **Manual** — a single pass using the **Loss Function**, **Categorical**, **Affine Only**, and **Parameters** you choose. Use this for label maps (Dice) or to run affine-only / a single custom loss.

> **Note:** registration currently runs on Slicer's main thread, so the user
> interface is unresponsive while a run is in progress. Progress is printed to
> the **Log** section; wait for "Registration complete" before interacting with
> Slicer again.

### Output Transform

Each registration run creates a new grid transform named, e.g., `NITorch_001_lcc_fixed_to_moving` (with incrementing counter and loss name). Use the **Output Transform** selector to switch between previously computed transforms.

### Parameters

**Lambda** (the overall nonlinear-regularization strength, default `10.0`) is a top-level control on the Registration tab — always editable, in both Automatic and Manual modes. The **Parameters** section below exposes the remaining settings; the effective nonlinear regularization for each term is `Lambda × Lambda <term>`.

| Parameter | Default | Description |
|---|---|---|
| Affine Basis | similitude | Degrees of freedom (translation, rotation, rigid, similitude, affine) |
| Affine Max Iter | 128 | Max iterations for affine optimizer per pyramid level |
| Affine Tolerance | 0.0001 | Convergence tolerance for affine optimizer |
| Global Max Iter | 64 | Max iterations for interleaved affine/nonlinear optimization |
| Global Tolerance | 0.0010 | Convergence tolerance for outer loop |
| Lambda Absolute | 0.0001 | Penalty on absolute displacements (0th order) |
| Lambda Membrane | 0.0010 | Penalty on membrane energy (1st order) |
| Lambda Bending | 0.2000 | Penalty on bending energy (2nd order) |
| Lambda Lame | 0.05, 0.2 | Lame constants for linear elastic energy (two comma-separated values) |
| Nonlin Max Iter | 64 | Max iterations for nonlinear optimizer per pyramid level |
| Nonlin Tolerance | 0.0010 | Convergence tolerance for nonlinear optimizer |
| Pyramid Levels | 0, 1, 2 | Coarse-to-fine pyramid levels (0 = finest) |

### Log

The **Log** section shows real-time registration progress including loss values and iteration counts.

### Visualization

The **Visualization** section provides tools for comparing fixed and moving volumes after registration. Controls are grayed out until both fixed and moving volumes are selected.

- **Enable comparison view** — overlays fixed (background) and moving (foreground) in the slice views with alpha blending. Links slice views and enables the crosshair. Unchecking restores the previous view layout.
- **Contours** — overlays iso-contour edges of the moving image on the fixed image (red outlines)
- **Opacity** slider — controls the blend ratio (0 = fixed only, 100 = moving only)
- **Contour Density** slider — controls the number of iso-contour levels (enabled when Contours is checked)

### Validation Tab

The **Validation** tab computes Dice overlap scores between fixed and moving segmentations:

1. Select a **Fixed** reference volume (typically the fixed image from registration)
2. Select **Fixed Labels** and **Moving Labels** (segmentation nodes)
3. Optionally select an **Output Transform** to evaluate (grid transform from registration)
4. Click **Compute Dice**

The **Summary** table shows the mean Dice (Before and After) for each computed result. Use the **Result** selector to view per-label Dice scores for a specific run. The **Mean Dice** label shows the selected result's mean. Green highlighting indicates improvement, red indicates degradation.

Results accumulate across runs — select different transforms and click Compute Dice again to compare multiple registrations. Use **Clear Results** to reset.
