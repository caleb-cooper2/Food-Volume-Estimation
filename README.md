# Volume Estimation

A pipeline server for processing and estimating the volume of food images taken via a mobile application.

## Approaches

Three approaches have been investigated, each as its own endpoint:

- **Monocular geometric** (`approaches/monocular.py`) -> single RGB image. Metric depth (DepthPro) + food mask (SAM 3), a support plane fitted to the surface the food sits on, then the volume integrated as a height field above that plane. Densities from the NLP server turn that volume into a mass, and its nutrients get rescaled to that mass
- **Deep learning** (`approaches/deep_learning.py`) -> single RGB image through the ConvNeXt-Tiny model trained on Nutrition5k (see [Training](#training-the-volume-estimation-model)). Predicts mass directly
- **Multi-view** (`approaches/multiview.py`) -> several RGB images reconstructed with VGGT, scale-anchored to a reference utensil measured in the reconstruction (falling back to DepthPro's metric depth over the background), then a watertight mesh volume per food instance (or the fused multi-view blob when per-instance meshing collapses)

## Prerequisites
- Python 3.10+
- CUDA-capable GPU strongly recommended (falls back to MPS on Apple Silicon, then CPU)
- A Hugging Face account with access to `facebook/sam3` for inference server (see note below)

## Getting started

### Clone the Repository
```bash
git clone <repository-url>
cd volume-estimation
```

### Create a Virtual Environment and Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

VGGT is not on pip, so the multi-view approach needs one more install straight from [their repo](https://github.com/facebookresearch/vggt):
```bash
pip install git+https://github.com/facebookresearch/vggt.git
```

---

## Inference Server

### Authenticate with Hugging Face
Both models are downloaded automatically from Hugging Face on first run. `apple/DepthPro-hf` is publicly available and requires no special access. `facebook/sam3` is a **gated model**, meaning you need to request and receive access approval from Meta before the weights can be downloaded.

**Step 1: Request access**

Visit [https://huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3), sign in, and submit an access request.

**Step 2: Generate an access token**

Once approved, go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and create a new token with at least **Read access to contents of all public gated repos you can access**.

**Step 3: Authenticate**

Login via the Hugging Face CLI (recommended, persists across sessions):

```bash
hf auth login
```

Or set the token as an environment variable for the current session:

```bash
export HF_TOKEN=hf_your_token_here
```

### Run the Server
`main:app` serves the monocular-geometric and deep-learning endpoints.

```bash
source .venv/bin/activate  # if not already active
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

The API will be available at `http://localhost:8001`. On first run, both models (~4.5GB total) will be downloaded and cached to `~/.cache/huggingface/`.

The deep-learning endpoint additionally needs a trained checkpoint at the path `approaches/deep_learning.py` registers as `DL_CHECKPOINT` (see [Training](#training-the-volume-estimation-model)).

### Run the Multi-View Server
The multi-view route attaches itself to the same app, so serving `approaches.multiview:app` gives you **all three** approaches on one port. Run this one instead of `main:app` when you want the multi-view endpoint, it only needs the extra VGGT install.

```bash
uvicorn approaches.multiview:app --reload --host 0.0.0.0 --port 8001
```

VGGT-1B (~5GB) is downloaded from Hugging Face on first run.

### Connect the NLP Server
The monocular endpoint calls the `unstructured-food-input` server to turn the user's text into food names (which prompt SAM 3 per item), densities (which turn volume into mass), and nutrients (which get rescaled from the text-implied grams to the photo-estimated `mass_g`). It's optional, without it the endpoint segments a single generic `food` mask and returns volume only.

It's expected on `http://localhost:8000`, override with `NLP_URL` if it's somewhere else

---

## Training the Volume Estimation Model

The training pipeline fine-tunes a ConvNeXt-Tiny regression model on the Nutrition5k dataset to predict food **mass** (in g) from a single overhead RGB image. Mass is the honest target here and matches the Nutrition5k baselines; `--target volume_density` can instead regress a density-derived volume. By default the target is regressed in log space (`--log_target`) since it's heavily skewed.

Optionally, `--use_volume` turns on volume-assisted regression: a cached geometric volume scalar (the same one the `/estimate-volume` endpoint computes) is concatenated into the regression head, so the model gets a real-world size cue alongside the RGB features. This needs the scalar cache built first, see [Volume-assisted regression](#volume-assisted-regression).

Training runs in two phases:

- **Phase 1** (epochs 1 -> `warmup_epochs`): backbone frozen, only the regression head is trained using ImageNet features as a fixed extractor
- **Phase 2** (epochs `warmup_epochs+1` -> end): backbone unfrozen with a 10x lower learning rate than the head, preventing the freshly initialised head from corrupting pretrained features

### Dataset (Nutrition5k)
The training pipeline uses the overhead RGB images, optionally side angles and dish metadata from the [Nutrition5k dataset](https://github.com/google-research-datasets/Nutrition5k).

A [Kaggle mirror](https://www.kaggle.com/datasets/gillesokhin/nutrition5k-dataset) is available with side angles pre-processed to sets of .jpeg images which makes life easier and takes less space.

The required layout expected under `--data_root`:

```
data/nutrition5k_dataset/
├── imagery/
│   ├── realsense_overhead/<dish_id>/rgb.png        # + depth_raw.png, needed for --depth_channel
│   └── side_angles/<dish_id>/camera_Aframe001.jpeg
└── ...
data/dish_metadata.csv                        # passed separately via --metadata
```

### Run Training

Each run takes a single metadata CSV. Run once per cafe, or concatenate the two CSVs before training.

```bash
# Train on cafe 1 (default)
python train.py \
  --data_root ./data/nutrition5k_dataset \
  --metadata ./data/dish_metadata_cafe1.csv \
  --output ./checkpoints \
  --epochs 50 \
  --warmup_epochs 5 \
  --batch 16 \
  --lr 1e-4

# Train on cafe 2
python train.py \
  --data_root ./data \
  --metadata ./data/dish_metadata_cafe2.csv \
  --output ./checkpoints
```

The best checkpoint (by validation MAPE) is saved to `checkpoints/best_model.pt`. Point `DL_CHECKPOINT` in `approaches/deep_learning.py` at whichever checkpoint you want served.

**All arguments**

| Argument          | Default                          | Description                                                                                    |
|-------------------|----------------------------------|------------------------------------------------------------------------------------------------|
| `--data_root`     | `./data/nutrition5k_dataset`     | Root directory containing `imagery/` and metadata CSVs                                         |
| `--metadata`      | `./data/dish_metadata_cafe1.csv` | Dish metadata CSV to train on                                                                  |
| `--output`        | `./checkpoints`                  | Directory to save best checkpoint                                                              |
| `--epochs`        | `50`                             | Total training epochs                                                                          |
| `--warmup_epochs` | `5`                              | Epochs to train head-only before unfreezing backbone                                           |
| `--batch`         | `16`                             | Batch size                                                                                     |
| `--lr`            | `1e-4`                           | Head learning rate (backbone uses `lr × 0.1` in phase 2)                                       |
| `--img_size`      | `224`                            | Input image size in pixels                                                                     |
| `--workers`       | `4`                              | DataLoader worker processes                                                                    |
| `--target`        | `mass`                           | Regression target: `mass` or `volume_density`                                                  |
| `--density`       | `0.8`                            | Density (g/cm³) used only by `--target volume_density`                                         |
| `--log_target`    | `True`                           | Regress in log space (use `--no-log_target` to disable)                                        |
| `--tta`           | `True`                           | Horizontal-flip test-time augmentation at val/test                                             |
| `--use_volume`    | `False`                          | Concat the cached geometric volume scalar into the head                                        |
| `--volume_cache`  | `./data/volume_scalars.csv`      | Volume-scalar cache CSV, required when `--use_volume`                                          |
| `--depth_channel` | `False`                          | Add Nutrition5k's overhead sensor depth as a 4th (relief) input channel (overhead dishes only) |

### Volume-assisted regression

Volume-assisted regression (`--use_volume`) feeds the geometric volume scalar into the model as an extra input. The scalar has to be cached per dish first — `cache_volume_scalars.py` runs the same SAM 3 -> DepthPro -> plane-fit -> height-integration pipeline as the `/estimate-volume` endpoint over every overhead image, so the training scalar and the deployed scalar come out of identical code.

```bash
# Build the per-dish volume scalars (writes ./data/volume_scalars.csv)
python cache_volume_scalars.py \
  --data_root ./data/nutrition5k_dataset \
  --out ./data/volume_scalars.csv

# Then train with the scalar concatenated into the head
python train.py \
  --metadata ./data/dish_metadata_cafe1.csv \
  --use_volume \
  --volume_cache ./data/volume_scalars.csv
```

| Argument       | Default                              | Description                                      |
|----------------|--------------------------------------|--------------------------------------------------|
| `--data_root`  | `./data/nutrition5k_dataset`         | Same root as `train.py`, contains `imagery/`     |
| `--out`        | `./data/volume_scalars.csv`          | Output CSV of per-dish volume scalars            |
| `--limit`      | `0`                                  | Cap dishes processed (0 = all), for a smoke test |
| `--log_every`  | `50`                                 | Progress log interval                            |

### Size prior
The `size_prior` scale anchor is a small MLP on frozen CLIP features that predicts the food's real-world footprint in cm, which is what makes the monocular depth metric without a utensil in frame. Same two-step shape as above: cache the targets, then train the head.

```bash
python cache_size_prior.py --data_root ./data/nutrition5k_dataset --out ./data/footprint_sizes.csv
python train_size_prior.py --footprint_cache ./data/footprint_sizes.csv --output ./checkpoints
```

---

## Benchmarking

`benchmark.py` drives the deployed endpoints over a manifest of labelled images, so it measures the same path the phone would use. Build a manifest, then run it:

```bash
python benchmark.py build-simplefood45 --labels ./data/ordered_dataset/labels.csv
python benchmark.py run --manifest ./data/benchmark_manifest_simplefood45.csv
```

It expects the server on `http://localhost:8001` (override with `VOLUME_API_URL`), and serving `approaches.multiview:app` since the manifest can ask for all three approaches. Results are written per row so an interrupted run resumes, and the summary tables break error down by approach, by arm (which scale anchor, text vs no text) and by camera pose.

---

## API

Every endpoint returns the same top-level structure, whichever approach produced it. Only what the approach actually predicts gets filled, the rest is `null`, and anything approach-specific sits under `diagnostics`:

```json
{
  "approach": "monocular-geometric",
  "volume_cm3": 312.4,
  "mass_g": 248.9,
  "confidence": "high",
  "diagnostics": { }
}
```

- `approach` -> `monocular-geometric` | `deep-learning` | `multi-view`
- `confidence` -> `low` | `medium` | `high`

### `POST /api/v1/estimate-volume`
Monocular geometric. `multipart/form-data`:

- `file` -> a JPEG, PNG or HEIC image (1KB - 30MB)
- `scale_ref` -> `utensil` (default) | `size_prior` | `checkerboard` | `auto`
- `text` -> optional, what the user typed. Goes to the NLP server for the per-item SAM 3 prompts and densities

`mass_g` is only filled for the items the NLP server had a density for, so it's `null` without `text`. `nutrients` follows the same rule: it's only filled where `mass_g` is.

```json
"diagnostics": {
  "food_pixel_count": 184203,
  "food_coverage_pct": 14.8,
  "plate_depth_m": 0.412,
  "intrinsics_source": "depthpro_fov",
  "scale_ref": "utensil",
  "scale_source": "utensil",
  "scale_factor": 1.083,
  "items_with_masses": 2,
  "items_with_nutrients": 2,
  "total_nutrients": {
    "energy_kj": 1084.3,
    "protein_g": 6.8,
    "fat_g": 1.2,
    "carbs_g": 58.4,
    "fibre_g": 3.1,
    "sodium_mg": 412.0
  },
  "nlp_available": true,
  "items": [
    {
      "prompt": "rice",
      "matched_food": "Rice, white, boiled",
      "volume_cm3": 186.2,
      "mass_g": 152.7,
      "mass_interval_g": [128.4, 181.6],
      "density_source": "measured",
      "presentation": null,
      "nutrients": {
        "energy_kj": 891.1,
        "protein_g": 4.9,
        "fat_g": 0.5,
        "carbs_g": 47.3,
        "fibre_g": 1.6,
        "sodium_mg": 305.4
      },
      "coverage_pct": 9.1,
      "segmentation_score": 0.881,
      "geometry_confidence": 0.94
    }
  ]
}
```

- `scale_source` -> the anchor that actually got used, `none` when they all failed and the raw depth was used as-is
- `total_nutrients` -> sum of `items[*].nutrients` across items that have one, `null` when none do 

### `POST /api/v1/estimate-volume-dl`
Deep learning. Same single-`file` request. Fills `mass_g` only, with `diagnostics` empty. Needs a trained checkpoint.

### `POST /api/v1/estimate-volume-multiview`
Multi-view, served by `approaches.multiview:app`. `multipart/form-data` with a `files` field of up to 10 images and the same optional `scale_ref` (`checkerboard` when benchmarking). Fills `volume_cm3` only

```json
"diagnostics": {
  "volume_source": "instance",
  "volume_instance_cm3": 298.7,
  "volume_blob_cm3": 341.2,
  "scale": 0.0417
}
```

- `volume_source` -> `instance` | `blob_fallback`, which of the two volumes above was returned
