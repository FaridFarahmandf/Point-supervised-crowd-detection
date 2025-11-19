# 🚀 Fast Training on UCF Mini Dataset (Lightweight Version)

Because the original LUDA pipeline and detector require very large GPU memory, we provide an alternative **Fast Training Mode** designed for quick experimentation on small subsets of the UCF dataset (e.g., 20 images).
This version is ideal for academic demonstrations, debugging, or low-resource training.

---

## 📂 1. Prepare the Mini Dataset

Place your reduced training and testing splits here:

```
data/UCF/TrainMini/
data/UCF/TestMini/
```

Each image must have a matching annotation file:

```
img_0001.jpg
img_0001_ann.mat
```

---

## ⚙️ 2. Generate LUDA Pseudo-Size Labels (Auto Version)

Run the LUDA auto-generator:

```bash
python LUDA_generate_UCF_AUTO.py
```

This produces:

```
data/cache/UCF/train/train.pkl
data/cache/UCF/test/test.pkl
```

These files contain:

* image paths
* point annotations
* LUDA pseudo bounding boxes
* pseudo-size weights

---

## 🧠 3. Train the Lightweight Detector

Use the simplified fast training script:

```bash
python train_UCF_fast.py
```

This version:

* uses ResNet-50 as backbone
* predicts two maps:

  * **center heatmap**
  * **pseudo-size map**
* trains on 256×256 resized images
* uses an 8×8 output resolution (fast, memory-safe)
* runs in **seconds per epoch** on a GPU

Output example:

```
Epoch 1/5
20/20 ━━━━━━━━━━━━━━ loss: XX.xx - pseudo_size_loss: XXX.xx
...
📊 Evaluation Results:
GT total: 2536
Pred total: 0
MAE: 507.2

💾 Model saved to: output_fast/ucf_fast_model.h5
```

⚠️ **Note:**
Fast training is intended for demonstration only — performance will be far below the full LUDA model.

---

## 👁️ 4. Visualize Predictions on Test Images

You can visualize heatmaps, overlays, and predicted points using:

```bash
python visualize_UCF.py
```

This produces:

```
visualizations/
    000_input.jpg
    000_heatmap.jpg
    000_overlay.jpg
    000_points.jpg
```

You will obtain:

* original image
* heatmap visualization
* heatmap overlay
* GT vs predicted points

These visualizations are excellent for inclusion in academic reports.

---

## 📦 5. Output Files

After training, the following structure is produced:

```
output_fast/
    ucf_fast_model.h5           # trained model (lightweight)
data/cache/UCF/
    train/train.pkl             # LUDA cache for training
    test/test.pkl               # LUDA cache for testing
visualizations/
    *.jpg                        # qualitative results
```

---

## 📝 Notes

* This fast pipeline is **not intended for state-of-the-art performance**.
* It serves as a **GPU-friendly alternative** to show

  * LUDA preprocessing
  * training procedure
  * inference
  * visualization
* Useful for coursework, demos, and debugging.


