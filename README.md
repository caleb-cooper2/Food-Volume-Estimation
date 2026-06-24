# Volume Estimation

A pipeline server for processing and estimating the volume of food images taken via a mobile application.

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
```bash
source .venv/bin/activate  # if not already active
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. On first run, both models (~4.5GB total) will be downloaded and cached to `~/.cache/huggingface/`.

---

## Training Volume Estimation Model

The training pipeline fine-tunes a ConvNeXt-Tiny regression model on the Nutrition5k dataset to predict food volume (in cm³) from a single overhead RGB image.

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
  --data_root ./data \
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

| Argument          | Default                          | Description                                              |
|-------------------|----------------------------------|----------------------------------------------------------|
| `--data_root`     | `./data`                         | Root directory containing `imagery/` and metadata CSVs   |
| `--metadata`      | `./data/dish_metadata_cafe1.csv` | Dish metadata CSV to train on                            |
| `--output`        | `./checkpoints`                  | Directory to save best checkpoint                        |
| `--epochs`        | `50`                             | Total training epochs                                    |
| `--warmup_epochs` | `5`                              | Epochs to train head-only before unfreezing backbone     |
| `--batch`         | `16`                             | Batch size                                               |
| `--lr`            | `1e-4`                           | Head learning rate (backbone uses `lr × 0.1` in phase 2) |
| `--img_size`      | `224`                            | Input image size in pixels                               |
| `--workers`       | `4`                              | DataLoader worker processes                              |

---

## API
### `POST /api/v1/estimate-volume`
Accepts a `multipart/form-data` request with a single `file` field containing a JPEG, PNG, or HEIC image (1KB - 30MB).
