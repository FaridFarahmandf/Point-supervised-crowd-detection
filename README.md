# 📌 **Point-Supervised Crowd Detection — UCF Dataset Adaptation**

This repository adapts the original implementation from
**“A Self-Training Approach for Point-Supervised Object Detection and Counting in Crowds”**
➡️ GitHub source: [https://github.com/WangyiNTU/Point-supervised-crowd-detection](https://github.com/WangyiNTU/Point-supervised-crowd-detection)

The goal of this project is to **train the LUDA-based point-supervised detector on a new crowd dataset** (UCF) that uses a **different annotation format** than the original WIDERFace dataset.

---

## 🎯 **Overview**

The original paper uses *single-file annotations*, where all image annotations are stored in one text file.
In our dataset (UCF), **each image has its own separate annotation file**.

Therefore, we modified the preprocessing and training pipeline to support this new annotation format.

This repository includes:

* ✔ Adaptation of LUDA preprocessing for the UCF dataset
* ✔ Custom loader for per-image annotation files
* ✔ Training script for UCF
* ✔ Evaluation script for UCF
* ✔ Same self-training steps as original: LUDA → refinement → detection

---

# 📂 **Dataset Structure**

Before running the code, ensure your dataset is placed as follows:

```
data/
 └── UCF/
      ├── Train/
      │     ├── img_001.jpg
      │     ├── img_001.txt
      │     ├── img_002.jpg
      │     ├── img_002.txt
      │     └── ...
      └── Test/
            ├── img_301.jpg
            ├── img_301.txt
            └── ...
```

* Each `.jpg` image must have a corresponding `.txt` annotation file.
* Annotation files contain **point coordinates only**.

---

# 🚀 **How to Run**

## **1️⃣ Step 1 — Download Dataset**

Download your UCF dataset (train + test) and place it inside:

```
data/UCF/Train/
data/UCF/Test/
```

---

## **2️⃣ Step 2 — Generate LUDA Cache Files**

Run the LUDA preprocessing script:

```bash
python LUDA_generate_UCF.py
```

This script:

* Reads all UCF image + point annotation pairs
* Generates LUDA-style pseudo sizes
* Saves processed data inside: `./cache/UCF/`

---

## **3️⃣ Step 3 — Train the Model on UCF**

```bash
python train_UCF.py
```

This will:

* Load LUDA preprocessed data
* Perform the self-training pipeline
* Save checkpoints into `./checkpoints/UCF/`

---

## **4️⃣ Step 4 — Evaluate on Test Set**

```bash
python test_UCF.py
```

This evaluates:

* Detection
* Point-localization
* Counting accuracy

---

# 📊 **Notes About Annotation Format**

| Original WIDERFace                      | UCF Dataset                                 |
| --------------------------------------- | ------------------------------------------- |
| Single annotation file                  | One annotation per image                    |
| Coordinates + bounding box size         | Only point coordinates                      |
| Original code assumes single label file | We modified loader to parse per-image files |

Our modified code handles loading each `.txt` annotation file individually.

---

# 🧩 **Project Motivation**

The goal of this project is to validate whether the LUDA self-training approach still performs well when applied to:

* A **different dataset**
* With **different annotation format**
* And more sparse or diverse point labels

This helps analyze the generalization ability of point-supervised crowd detection models.

---

# 🙌 **Acknowledgements**

This project is based on the original implementation by:

**Wang, Yi et al. (NTU)**
[https://github.com/WangyiNTU/Point-supervised-crowd-detection](https://github.com/WangyiNTU/Point-supervised-crowd-detection)

We extend their codebase to support new datasets and annotation formats.

