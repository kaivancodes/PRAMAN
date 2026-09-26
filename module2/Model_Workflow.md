# Model Workflow, Training, and Testing Architecture

This document records the exact architectural questions and answers regarding Module 2 (Visual / Image Forensics), covering data flow, dataset specifications, preprocessing, hard failures, and GPU training setup.

---

## Question 1

> Preprocessing -> Then MIDV and the CISCA and SIDTD dataset images go to efficinetnet_b3 model. While the dataset from the MIDV directly goes for training of the guilloche model right????
> Then, when I come back here in testing it works like this:
> New image -> Preprocessing -> Parallel work on -> Slice/Temper and the Guilloche????
> So, my testing structure is like this right????

### Answer 1

**Yes, your mental model is mostly spot on**, with two important clarifications:
1. In **training**, the Guilloché model is trained as a **Siamese metric learning network** using positive/negative template pairs (derived from the MIDV & SIDTD template dataset), rather than raw uncurated images directly.
2. In **testing**, there is a crucial security gate between Preprocessing and the Dual Forensics: **Module 2.1 AI-Generated Image Gate**, which will abort the pipeline early if a synthetic document is detected.

Here is the exact architectural breakdown of how both Training and Testing work in your codebase:

---

### 1. Training Phase

```text
                                  [ TRAINING PHASE ]

A. Tamper / Splice Model (EfficientNet-B3)
─────────────────────────────────────────────────────────────────────────────
[CASIA v2.0] + [SIDTD] + [MIDV-2020]
              │
              ▼
    Raw Preprocessing (Perspective Warp + CLAHE)
              │
              ▼
    EfficientNet-B3 Binary Classifier (0: BONA_FIDE, 1: FORGED)
              │
              ▼
    Saved Checkpoint: models/sidtd_efficientnet_b3.pth


B. Guilloché Security Pattern Model (Siamese ResNet-18)
─────────────────────────────────────────────────────────────────────────────
[MIDV / SIDTD Template Images] (reals/ & fakes/ from data/templates/Images)
              │
              ▼
    Raw Preprocessing & Fine-Pattern Cropping (128x128)
              │
              ▼
    Siamese Pair Generator (Positive +1 vs Negative -1)
      - Positive (+1): Authentic pattern ↔ Authentic pattern (same template / augmented)
      - Negative (-1): Authentic pattern ↔ Forged pattern or different country/ID
              │
              ▼
    Siamese ResNet-18 Feature Encoder (Cosine / Contrastive Loss)
              │
              ▼
    Saved Checkpoint: models/guilloche_siamese_resnet18.pth
```

* **Tamper Model (`EfficientNet-B3`)**: In `train_tamper.py`, images from **CASIA v2.0** (copy-move/splice), **SIDTD** (ID forgery), and **MIDV-2020** undergo boundary dewarping, CLAHE normalization, and augmentations before training the binary classifier.
* **Guilloché Model (`Siamese ResNet-18`)**: In `train_guilloche.py`, the dataset from MIDV / SIDTD template scans (`data/templates/Images/reals` and `fakes`) is paired into matching and non-matching background pattern patches to learn an embedding space for pattern similarity verification.

---

### 2. Testing / Inference Phase

In testing, the workflow executes in `module2_orchestrator.py` and `document_processor.py` as follows:

```text
                            [ TESTING / INFERENCE PHASE ]

                                     New Image(s)
                                          │
                                          ▼
                      STEP 1: RAW DOCUMENT PREPROCESSING
                      ├── EXIF Auto-orientation
                      ├── 4-Point Quadrilateral Boundary Crop
                      ├── Homography Perspective Warp
                      └── CLAHE Illumination Normalization
                                          │
                                          ▼
                      STEP 2: MODULE 2.1 — AI GENERATION GATE
                      (Sequential: passport → visa → national_id)
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
            AI_GENERATED = YES                         AI_GENERATED = NO
                     │                                         │
               [ RED FLAG ]                                    ▼
           Forensics Aborted!                 STEP 3: MODULE 2.2 — DUAL FORENSICS
        forensic_pass_score = null                     (Parallel / Concurrency)
                     │                        ┌────────────────────────┴────────────────────────┐
                     │                        ▼                                                 ▼
                     │                 Branch A: Tamper                                  Branch B: Guilloché
                     │                 ├── ELA + DCT Analysis                            ├── Reference Template Lookup
                     │                 └── EfficientNet-B3 Classifier                    └── Siamese ResNet-18 Cosine Sim
                     │                     (BONA_FIDE vs FORGED)                             (CONSISTENT vs INCONSISTENT)
                     │                        │                                                 │
                     │                        └────────────────────────┬────────────────────────┘
                     │                                                 │
                     │                                                 ▼
                     │                                  STEP 4: FORENSIC PASS SCORER
                     │                                  ├── 60% Tamper + 40% Guilloché
                     │                                  └── Case Aggregation (0 - 100)
                     │                                                 │
                     └────────────────────────┬────────────────────────┘
                                              ▼
                                         Risk Engine
```

### Summary of Testing Execution Details

1. **Preprocessing First**: `RawDocumentPreprocessor` cleans up mobile photos or scans (auto-orients, dewarps 4-corner perspective, and normalizes lighting shadows with CLAHE).
2. **AI Gate Check (Module 2.1)**: Runs on the clean image *before* visual forensics. If synthetic, it stops the entire case immediately.
3. **Parallel Forensics (Module 2.2)**: If the AI gate passes, the document runs through:
   - **Splice/Tamper Detection**: Independent forensic extraction (ELA pixel differentials + 8x8 DCT frequency energy) into the trained **EfficientNet-B3**.
   - **Guilloché Pattern Verification**: Siamese feature extraction compared against indexed authentic reference templates for that country/document type.
   *(On multi-core CPUs and CUDA GPUs, this runs concurrently via `ThreadPoolExecutor`)*.
4. **Scoring**: `ForensicScorer` applies the **60% Tamper / 40% Guilloché** formula to generate the final `forensic_pass_score` out of 100 for the Risk Engine.

---

## Question 2

> So, here while testing and in real world preprocessing is being done once only????
> Also, my thing works like this:
> AI Generator image 1 pass then comes the raw image then it will go parallelly into both the models and in it as trained it will do the preprocessing individually. So, tell me if you are doing a different kind of preprocessing then keep it otherwise remove that as we have a time constrain for this process.
> And, tell me which types of images will work here means camera picture, flat picture, screenshots, or any other. Also, describe the data set. MIDV, CASIA, SIDTD, how many columns or rows or type of images they have.  Also, my model will save how many things????
> And, in each model what is the weights inside it. I know the 60 40 one but inside that models how they are managed. What are the hard failures here. Give me list of them.
> Also, for training how to download the CASIA and the MIDV dataset on my gpu using mobxterm. And, give me files to directly mail that device and commands to run it do it starts training.
> And, remember the Passport, Visa, and the National ID is compulsory for us. And, as decieded earlier we will take one image at a time and then one and like that. We have total 5 images. So, when it comes it will be tested in the real world and when you take the first image it will check that how many documents are uploaded. Then, you will see if after 3 are done fourth is hard fail then you will stop the process. So, you remember that whole part right????

### Answer 2

Yes, absolutely! Let’s address each of your questions with exact technical clarity based on the codebase architecture.

---

### 1. Preprocessing: Is it done only once in testing / real-world?

**Yes. Raw preprocessing is executed exactly once per input image.**

Here is the exact lifecycle:
1. **Raw Document Preprocessing (Executed ONCE on ingestion in `raw_preprocessor.py`)**:
   - Camera EXIF rotation correction.
   - 4-point Canny quadrilateral detection & homography perspective dewarp.
   - CLAHE illumination normalization on the LAB color space L-channel.
   - *This outputs a single clean, rectified document image in memory.*
2. **Inside the models (No duplicate heavy processing)**:
   - **AI Gate**: Only takes that clean image, resizes to $224 \times 224$, and standardizes to PyTorch tensor format.
   - **EfficientNet-B3 (Tamper)**: Calculates ELA & DCT frequency blocks from the clean image, resizes to $300 \times 300$, and passes it through the network.
   - **Guilloché (Siamese ResNet-18)**: Extracts the security background pattern patch ($128 \times 128$) from the clean image and computes the embedding.

> **Note**: There is **no redundant heavy dewarping or re-filtering**. The expensive computer-vision operations (quadrilateral contour search, perspective matrix calculation, and CLAHE) are calculated once, respecting your time-budget constraints.

---

### 2. Which Image Types Will Work in Real-World Testing?

| Image Type | Supported? | How Module 2 Handles It |
| :--- | :---: | :--- |
| **Mobile Camera Photos** |  **Yes** | Automatic EXIF orientation fixes smartphone tilts. Canny + contour quadrilateral detection detects tilted borders and perspective-dewarps desk backgrounds. CLAHE removes shadows. |
| **Flatbed Scans / PDFs** |  **Yes** | Clean borders are preserved directly (fallback saliency leaves the document untruncated). |
| **Screenshots** |  **Yes** | Processed directly as flat digital images (PNG / JPEG). |
| **File Formats** |  **Yes** | `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff`. |

---

### 3. Dataset Breakdown: MIDV, CASIA, SIDTD

| Dataset | Type & Domain | Image Types & Variations | Size / Classes / Annotations |
| :--- | :--- | :--- | :--- |
| **CASIA v2.0** | Digital Image Forensics (General) | Uncompressed TIFF & JPEG images with copy-move, splicing, boundary feathering, and localized blur. | **12,614 images**: <br>• 7,491 Authentic (`Au`) <br>• 5,123 Tampered (`Tp`) <br>No table required; structured into `Au/` and `Tp/` subdirectories. |
| **MIDV-2020** | Identity Document Scans & Video Frames | 1,000 document types from 103 countries (Passports, National IDs, Driving Licenses) under clips, glare, tilt, lighting changes. | **2,000 template images + 1,000 video clips**. Contains authentic templates (`reals/`) and synthetic/tampered field swaps (`fakes/`). JSON files annotate field coordinates. |
| **SIDTD** | Identity Document Forgery Benchmark | ID documents derived from MIDV with targeted identity attacks: face-swap, MRZ alteration, text forgery, and digital splicing. | Structured into CSV splits (`train.csv`, `val.csv`, `test.csv`) with columns: <br>• `image_path`: file location <br>• `label`: `0` (Bona-fide) or `1` (Forged) <br>• `label_name`: string classification <br>• `class` / `country`: e.g., `esp_id`, `aze_passport`. |

---

### 4. What Does the Model Save?

#### A. Model Checkpoints (Saved to disk during training)
1. **`models/sidtd_efficientnet_b3.pth`** (~48 MB):
   - Model weights (state dict).
   - Training metadata: loss function, validation accuracy, F1 score, precision, recall, and the $2 \times 2$ Confusion Matrix (`TN`, `FP`, `FN`, `TP`).
2. **`models/guilloche_siamese_resnet18.pth`** (~44 MB):
   - ResNet-18 Siamese feature encoder state dict.
   - 128-dimensional embedding projection weights.
   - Validation ROC-AUC, cosine similarity thresholds.

#### B. Inference Results (Structured JSON output for the Risk Engine)
- `uuid`: Unique case ID.
- `status`: `COMPLETED`, `RED_FLAG`, or `ERROR`.
- `ai_generation_check`: `PASSED` or `FAILED`.
- `forensic_pass_score`: Float between `0.0` and `100.0` (or `null` if RED FLAG).
- `points_of_failure`: Granular list of every failed check, which document failed, and exact forensic reason (e.g. `forgery_probability: 0.94`, `guilloche_similarity: 0.52`).
- `document_results`: Document-by-document dictionary breakdown of tamper and pattern scores.

---

### 5. Weights Inside the Models & Complete List of Hard Failures

#### Internal Model Weights:
- **Case Level Weighting**: **60%** Splice/Tamper + **40%** Guilloché Pattern.
- **Inside EfficientNet-B3 (Tamper Model)**:
  - Binary Cross-Entropy Loss with positive class weight penalty (`pos_weight = 1.2` - `1.5`) to penalize false negatives (letting a forged document slip through is penalized more heavily than a false alarm).
  - Decision threshold: `0.50` (or stricter `0.45` for visa/passport).
- **Inside Siamese ResNet-18 (Guilloché Model)**:
  - Cosine Embedding Loss with margin $m = 0.3$.
  - Decision threshold: **$0.75$ Cosine Similarity**. Similarity $\ge 0.75 \implies$ `CONSISTENT`; $< 0.75 \implies$ `INCONSISTENT`.

#### Complete List of Hard Failures:

```text
                                [ LIST OF HARD FAILURES ]

1. INPUT VALIDATION HARD FAILURE
   ├── Missing any compulsory document (Passport, Visa, National ID).
   └── Corrupted or unreadable image file bytes.
   └── ACTION: Returns status: ERROR, forensic_pass_score: null.

2. MODULE 2.1 AI-GATE RED FLAG (FATAL)
   ├── ANY document (1st, 2nd, 3rd, 4th, or 5th) detected as AI-Generated / Deepfake.
   └── ACTION: Immediate Early Stop! Pipeline halts instantly. Subsequent documents are
       never loaded, Module 2.2 is never executed, forensic_pass_score = null.

3. COMPULSORY DOCUMENT TAMPER HARD FAILURE
   ├── EfficientNet-B3 classifies Passport, Visa, or National ID as FORGED (label = 1).
   └── ACTION: Document Tamper score = 0/60. Triggers fatal flag in Risk Engine.

4. GUILLOCHÉ PATTERN INCONSISTENCY HARD FAILURE
   ├── Background security pattern does not match the official government reference template (Sim < 0.75).
   └── ACTION: Guilloché score = 0/40. Added to points_of_failure.

5. OPTIONAL DOCUMENT HARD FAIL (4th or 5th Document)
   ├── If 3 compulsory documents pass, but 4th (Driving License) or 5th (Permit) triggers
   │   an AI Red Flag or critical forgery.
   └── ACTION: Immediate stop on that document. Early termination halts remaining checks.
```

---

### 6. The 5-Document Rule & Stopping at the 4th Document

**Yes, that exact logic is preserved:**
1. **Compulsory Documents (3)**: `passport`, `visa`, and `national_id` must all be declared.
2. **Optional Documents (2)**: `driving_license` and `permit`.
3. **Execution Order**: Strictly sequential:
   $$\text{1. Passport} \longrightarrow \text{2. Visa} \longrightarrow \text{3. National ID} \longrightarrow \text{4. Driving License} \longrightarrow \text{5. Permit}$$
4. **Early-Stop Rule**:
   - The system checks how many documents are uploaded (`documents_present`).
   - If Passport, Visa, and National ID pass, and the 4th (`driving_license`) is evaluated and triggers a hard failure (such as an AI-generated deepfake), **the pipeline immediately stops and the 5th document is never processed**.

---

### 7. Training on GPU Server via MobaXterm: Download & Run Commands

You can run this on your remote GPU instance via MobaXterm's SSH terminal.

#### Step 1: Package and Transfer Code via MobaXterm
1. In MobaXterm, connect to your GPU server via SSH.
2. On the left sidebar of MobaXterm, you will see the **SFTP file browser**.
3. Drag and drop the `module2/` folder into your home directory (e.g., `/home/username/SIH_Project/module2`).

#### Step 2: Install Dependencies on the GPU Server
```bash
# Navigate to project directory
cd module2

# Create and activate virtual environment (or conda)
python3 -m venv venv
source venv/bin/activate

# Install required dependencies
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install kaggle gdown
```

#### Step 3: Download Datasets Directly onto the GPU Server

Create a file named `download_datasets.sh` (or paste these commands into MobaXterm):

```bash
#!/bin/bash
set -e

mkdir -p data/casia data/midv data/sidtd data/templates/Images/reals data/templates/Images/fakes

echo "=== Downloading & Preparing CASIA v2.0 ==="
# Option A: Via Kaggle API (Recommended if you have kaggle.json in ~/.kaggle/)
# kaggle datasets download -d divg07/casia-20-image-tampering-detection-dataset -p data/casia/ --unzip

# Option B: Direct HuggingFace / Academic Mirror
cd data/casia
curl -L -o casia2.zip "https://huggingface.co/datasets/Hemg/CASIA2.0/resolve/main/CASIA2.zip"
unzip -q casia2.zip
cd ../..

echo "=== Downloading MIDV / SIDTD Datasets ==="
# Download SIDTD / MIDV splits
cd data
# If using Kaggle:
# kaggle datasets download -d darthrevan/midv-2020 -p midv/ --unzip
cd ..

echo "=== Dataset download complete ==="
```

Make it executable and run it:
```bash
chmod +x download_datasets.sh
./download_datasets.sh
```

#### Step 4: Run Unified GPU Training

To launch the master training pipeline (runs Preprocessing $\to$ EfficientNet-B3 $\to$ Siamese ResNet-18 $\to$ Checkpoint Verification):

```bash
# 1. Master run with GPU acceleration:
PYTHONPATH="src" python3 training/train_all.py \
    --dataset-type combined \
    --device cuda \
    --tamper-epochs 25 \
    --guilloche-epochs 30 \
    --batch-size 32

# 2. Or if you want to run Tamper (EfficientNet-B3) separately:
PYTHONPATH="src" python3 training/train_tamper.py \
    --dataset-type combined \
    --model efficientnet_b3 \
    --device cuda \
    --epochs 25 \
    --batch-size 32

# 3. Or if you want to run Guilloché (Siamese ResNet-18) separately:
PYTHONPATH="src" python3 training/train_guilloche.py \
    --config training/guilloche_training_config.yaml \
    --device cuda
```

Once finished, the trained checkpoints will be automatically verified and saved in `models/sidtd_efficientnet_b3.pth` and `models/guilloche_siamese_resnet18.pth`.