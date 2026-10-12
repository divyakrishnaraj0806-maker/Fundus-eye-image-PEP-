# Fundus AI — Multi-Disease Retinal Screening Pipeline

An end-to-end deep-learning pipeline for retinal fundus images. From a single photo it produces:

1. **8-class multi-label disease classification** (DenseNet121)
2. **Diabetic Retinopathy (DR) severity grading**, grade 0–4 (DenseNet121)
3. **Lesion segmentation** (MA, HE, EX, SE) plus **optic disc / optic cup segmentation** (U-Net with DenseNet121 encoder)
4. **Vertical Cup-to-Disc Ratio (CDR)** for glaucoma risk
5. **Grad-CAM** explanations, a **consistency check** across models, and a final **CONFIDENT / DOCTOR REVIEW** decision with a 6-panel report image

## Classes

| Code | Condition | Code | Condition |
|---|---|---|---|
| N | Normal | H | Hypertensive retinopathy |
| D | Diabetic retinopathy | M | Myopia |
| G | Glaucoma | O | Other abnormalities |
| C | Cataract | A | Age-related macular degeneration |

DR severity grades: No DR · Mild · Moderate · Severe · Proliferative.

---

## Datasets (not included in this repo — download from Kaggle)

| Key | Used for | Kaggle source |
|---|---|---|
| `odir` | 8-class labels (ODIR-5K) | `andrewmvd/ocular-disease-recognition-odir5k` |
| `aptos` | DR grade 0–4 (APTOS 2019, 224×224) | `sovitrath/diabetic-retinopathy-224x224-2019-data` |
| `idrid` | DR grade + lesion masks + optic disc | `aaryapatel98/indian-diabetic-retinopathy-image-dataset` |
| `refuge` | Glaucoma label + optic disc / cup masks | `victorlemosml/refuge2` |
| *(weights)* | DenseNet121 ImageNet weights (offline) | `divyakdharsh/densenet121-pretrained-weights` |


Data used: **11,548 images** (ODIR 6,970 · APTOS 3,662 · IDRiD 516 · REFUGE 400),
split **70 / 20 / 10** (train / test / val) at **patient level** with class stratification, so both eyes of one patient never cross splits.

---

## Method summary

- **Preprocessing:** crop black border → resize 320×320 → ImageNet normalisation
- **Augmentation (train only):** flips, rotation ±20°, brightness/contrast jitter, random erasing
- **Classifier:** DenseNet121 (timm), `BCEWithLogitsLoss` with square-root `pos_weight` (clipped 1–6) for rare classes
- **Training:** AdamW (lr 3e-4, wd 1e-4), cosine schedule, batch 24, mixed precision, early stopping (patience 8), resume/warm-start checkpoints
- **Thresholds:** per-class thresholds tuned on the validation set
- **DR grading:** separate DenseNet121 (5-way), evaluated with Quadratic Weighted Kappa
- **Segmentation:** U-Net with DenseNet121 encoder, thresholds tuned on validation

---

## Results (held-out test set)

### Disease classification (2,310 images)

| Class | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| N | 0.785 | 0.892 | 0.835 | 0.923 |
| D | 0.894 | 0.803 | 0.846 | 0.946 |
| G | 0.526 | 0.610 | 0.565 | 0.962 |
| C | 0.571 | 0.933 | 0.709 | 0.985 |
| A | 0.323 | 0.582 | 0.416 | 0.939 |
| H | 0.283 | 0.386 | 0.327 | 0.877 |
| M | 0.860 | 0.860 | 0.860 | 0.997 |
| O | 0.521 | 0.519 | 0.520 | 0.903 |

Overall: per-label accuracy **0.938**, exact-match **0.705**, macro P/R/F1 **0.596 / 0.698 / 0.635**, **macro ROC-AUC 0.941**.

### DR severity grading (840 images)

Accuracy **0.818** · QWK **0.854** · within-1-grade **0.924**
(APTOS: acc 0.845, QWK 0.878 · IDRiD: acc 0.625, QWK 0.698)

### Segmentation (Dice)

| Source | MA | HE | EX | SE | Optic disc | Optic cup |
|---|---|---|---|---|---|---|
| IDRiD | 0.390 | 0.641 | 0.705 | 0.591 | 0.969 | – |
| REFUGE | – | – | – | – | 0.963 | 0.893 |

Vertical CDR on REFUGE test: **MAE 0.039**, correlation **0.867**.

> Rare classes (A, H, G) and small lesions (MA) are the weak spots — see limitations below.

---

## Repository structure

```
fundus-ai/
├── notebooks/
│   └── final-project.ipynb      # full pipeline (Cells 1–15 + manual upload demo)
├── models/                      # trained weights (see "Model weights" below)
│   ├── best_model.pth           # 8-class classifier
│   ├── best_grade_model.pth     # DR grading
│   └── best_seg_model.pth       # lesion + disc/cup segmentation
├── configs/
│   ├── class_thresholds.json
│   └── seg_thresholds.json
├── results/
│   ├── figs/                    # confusion matrix, ROC, training curves, samples
│   ├── reports/                 # example 6-panel reports
│   └── *.csv                    # test_metrics_per_class, seg_test_metrics, logs
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How to run

### Option A — Kaggle (recommended, what this project was built on)

1. Upload `notebooks/final-project.ipynb` to Kaggle (**New Notebook → File → Import Notebook**).
2. Set **Accelerator → GPU (T4 / P100)**.
3. **Add Input** → attach all datasets listed in the table above (ODIR-5K, APTOS, IDRiD, REFUGE, DenseNet121 weights).
4. For inference only: upload the three `.pth` files as a Kaggle Dataset and copy them into `/kaggle/working/outputs/` (Cell 13 expects `best_seg_model.pth` there).
5. Run all cells. Outputs are written to `/kaggle/working/outputs/` and zipped as `fundus_ai_results.zip`.
6. To test a single image, run the last two cells (upload widget → full report).

### Option B — Colab / local

The notebook auto-detects the environment and downloads the public datasets via `kagglehub`. You need a Kaggle API token (`KAGGLE_USERNAME`, `KAGGLE_KEY`). A GPU is strongly recommended.

```bash
git clone https://github.com/<your-username>/fundus-ai.git
cd fundus-ai
pip install -r requirements.txt
jupyter notebook notebooks/final-project.ipynb
```

---

## Model weights

`.pth` files are large, so they are kept out of normal git history. Download them from:

- **GitHub Releases:** `<add link>`  (or)
- **Kaggle Dataset / Model:** `<add link>`

Place them in `models/` (or `/kaggle/working/outputs/` on Kaggle).

---

## Limitations

- Class imbalance: A, H and G have few samples, so precision/F1 are low for them.
- Small lesions (microaneurysms) are hard to segment (Dice ≈ 0.39).
- Severe and Proliferative DR recall is lower than other grades; IDRiD generalises worse than APTOS.
- Public datasets only; no external clinical validation.
- Output is a screening aid. Final decisions must be made by a qualified ophthalmologist.

---

## Acknowledgements

ODIR-5K, APTOS 2019, IDRiD, REFUGE (and optionally FGADR / DDR) dataset authors; `timm` and PyTorch.

## Author

**Divyasri K** — IT, St. Joseph's Institute of Technology, Chennai
