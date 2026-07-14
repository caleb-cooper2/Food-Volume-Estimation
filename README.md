# Volume Estimation

A pipeline server for processing and estimating the volume of food images taken via a mobile application.

## Approaches

Three approaches have been investigated, each as its own endpoint:

- **Monocular geometric** (`main.py`) -> single RGB image. Metric depth (DepthPro) + food mask (SAM 3), a support plane fitted to the surface the food sits on, then the volume integrated as a height field above that plane.
- **Deep learning** (`main.py`) -> single RGB image through the ConvNeXt-Tiny model trained on Nutrition5k (see [Training](#training-volume-estimation-model)). Predicts mass directly
- **Multi-view** (`multi-image.py`) -> several RGB images reconstructed with VGGT, scale-anchored to a reference utensil measured in the reconstruction (falling back to DepthPro's metric depth over the background), then a watertight mesh volume per food instance (or the fused multi-view blob when per-instance meshing collapses)

The first two live in `main.py` (they share the depth/segmentation/geometry core logic) while the multi-view route lives in `multi-image.py`. All three return the same [response structure](#api)

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
cd server
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
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
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. On first run, both models (~4.5GB total) will be downloaded and cached to `~/.cache/huggingface/`.

The deep-learning endpoint additionally needs a trained checkpoint at `checkpoints/best_model.pt` (see [Training](#training-volume-estimation-model)).

### Run the Multi-View Server
The multi-view route lives in its own app in `multi-image.py`. It uses [VGGT](https://github.com/facebookresearch/vggt), installation instructions provided in their repo.

```bash
uvicorn multi-image:app --reload --host 0.0.0.0 --port 8001
```

VGGT-1B (~5GB) is downloaded from Hugging Face on first run.

---

## Training Volume Estimation Model

The training pipeline fine-tunes a ConvNeXt-Tiny regression model on the Nutrition5k dataset to predict food **mass** (in g) from a single overhead RGB image. Mass is the honest target here and matches the Nutrition5k baselines; `--target volume_density` can instead regress a density-derived volume. By default the target is regressed in log space (`--log_target`) since it's heavily skewed.

Optionally, `--use_volume` turns on volume-assisted regression: a cached geometric volume scalar (the same one the `/estimate-volume` endpoint computes) is concatenated into the regression head, so the model gets a real-world size cue alongside the RGB features. This needs the scalar cache built first, see [Volume-assisted regression](#volume-assisted-regression).

Training runs in two phases:

- **Phase 1** (epochs 1 -> `warmup_epochs`): backbone frozen, only the regression head is trained using ImageNet features as a fixed extractor
- **Phase 2** (epochs `warmup_epochs+1` -> end): backbone unfrozen with a 10x lower learning rate than the head, preventing the freshly initialised head from corrupting pretrained features

### Dataset (Nutrition5k)
The training pipeline uses the overhead RGB images and dish metadata from the [Nutrition5k dataset](https://github.com/google-research-datasets/Nutrition5k).

```bash
mkdir -p data
```

#### Download overhead images only
This pulls only the `realsense_overhead` imagery directory (~3.5GB) rather than the full 181GB archive.

```bash
gcloud storage cp -r "gs://nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead" data/
```

#### Download the three metadata CSVs
```bash
gcloud storage cp \
  "gs://nutrition5k_dataset/nutrition5k_dataset/metadata/dish_metadata_cafe1.csv" \
  "gs://nutrition5k_dataset/nutrition5k_dataset/metadata/dish_metadata_cafe2.csv" \
  "gs://nutrition5k_dataset/nutrition5k_dataset/metadata/ingredients_metadata.csv" \
  data/
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

The best checkpoint (by validation MAPE) is saved to `checkpoints/best_model.pt`.

**All arguments**

| Argument          | Default                            | Description                                                                                    |
|-------------------|------------------------------------|------------------------------------------------------------------------------------------------|
| `--data_root`     | `./data/nutrition5k_dataset`       | Root directory containing `imagery/` and metadata CSVs                                         |
| `--metadata`      | `./data/dish_metadata_cafe1.csv`   | Dish metadata CSV to train on                                                                  |
| `--output`        | `./checkpoints`                    | Directory to save best checkpoint                                                              |
| `--epochs`        | `50`                               | Total training epochs                                                                          |
| `--warmup_epochs` | `5`                                | Epochs to train head-only before unfreezing backbone                                           |
| `--batch`         | `16`                               | Batch size                                                                                     |
| `--lr`            | `1e-4`                             | Head learning rate (backbone uses `lr × 0.1` in phase 2)                                       |
| `--img_size`      | `224`                              | Input image size in pixels                                                                     |
| `--workers`       | `4`                                | DataLoader worker processes                                                                    |
| `--target`        | `mass`                             | Regression target: `mass` or `volume_density`                                                  |
| `--density`       | `0.8`                              | Density (g/cm³) used only by `--target volume_density`                                         |
| `--log_target`    | `True`                             | Regress in log space (use `--no-log_target` to disable)                                        |
| `--tta`           | `True`                             | Horizontal-flip test-time augmentation at val/test                                             |
| `--use_volume`    | `False`                            | Concat the cached geometric volume scalar into the head                                        |
| `--volume_cache`  | `./data/volume_scalars.csv`        | Volume-scalar cache CSV, required when `--use_volume`                                          |
| `--depth_channel` | `False`                            | Add Nutrition5k's overhead sensor depth as a 4th (relief) input channel (overhead dishes only) |

### Volume-assisted regression

Volume-assisted regression (`--use_volume`) feeds the geometric volume scalar into the model as an extra input. The scalar has to be cached per dish first — `cache_volume_scalars.py` runs the same SAM 3 -> DepthPro -> plane-fit -> height-integration pipeline as the `/estimate-volume` endpoint over every overhead image, so the training scalar and the deployed scalar come out of identical code.

```bash
# Build the per-dish volume scalars (writes ./data/volume_scalars.csv)
python cache_volume_scalars.py \
  --data_root ./data/nutrition5k_dataset/imagery \
  --out ./data/volume_scalars.csv

# Then train with the scalar concatenated into the head
python train.py \
  --metadata ./data/dish_metadata_cafe1.csv \
  --use_volume \
  --volume_cache ./data/volume_scalars.csv
```

| Argument       | Default                              | Description                                      |
|----------------|--------------------------------------|--------------------------------------------------|
| `--data_root`  | `./data/nutrition5k_dataset/imagery` | Root of the overhead imagery directories         |
| `--out`        | `./data/volume_scalars.csv`          | Output CSV of per-dish volume scalars            |
| `--limit`      | `0`                                  | Cap dishes processed (0 = all), for a smoke test |
| `--log_every`  | `50`                                 | Progress log interval                            |

---

## API

Every endpoint returns the same structure, no matter the approach that generates it. `volume_cm3` or `mass_g` is populated depending on what the approach actually predicts, and anything approach-specific gets placed under `diagnostics`:

```json
{
  "approach": "monocular-geometric",
  "volume_cm3": 312.4,
  "mass_g": null,
  "confidence": "high",
  "diagnostics": { }
}
```

| Field         | Type              | Notes                                                             |
|---------------|-------------------|-------------------------------------------------------------------|
| `approach`    | string            | `monocular-geometric` \| `deep-learning` \| `multi-view`          |
| `volume_cm3`  | float \| null     | Estimated food volume, `null` when the approach predicts mass     |
| `mass_g`      | float \| null     | Estimated food mass, only the deep-learning approach fills this   |
| `confidence`  | string \| null    | `low` \| `medium` \| `high`, only where the approach computes one |
| `diagnostics` | object            | Approach-specific extras (see each endpoint below)                |

### `POST /api/v1/estimate-volume`
Monocular geometric. Accepts `multipart/form-data` with a single `file` field containing a JPEG, PNG, or HEIC image (1KB - 30MB). Fills `volume_cm3` and `confidence`; `diagnostics` carries `food_pixel_count`, `food_coverage_pct`, `max_food_height_cm`, `mean_food_height_cm`, `plate_depth_m`, `intrinsics_source` and a base64 `debug_overlay_b64` of the segmentation.

### `POST /api/v1/estimate-volume-dl`
Deep learning. Same single-`file` `multipart/form-data` request. Fills `mass_g` (`volume_cm3` and `confidence` are `null`, `diagnostics` empty). Requires a trained `checkpoints/best_model.pt`.

### `POST /api/v1/estimate-volume-multiview`
Multi-view (served by `multi-image:app`). Accepts `multipart/form-data` with a `files` field of up to 10 images and an optional `context` field carrying an NLP portion prior as a JSON string (e.g. `{"total_prior_cm3": 240, "total_prior_cv": 0.3}`). Fills `volume_cm3`; `diagnostics` carries `volume_source` (`instance` or `blob_fallback`), `volume_instance_cm3`, `volume_blob_cm3` and `scale`.
