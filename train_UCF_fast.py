import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import cv2
import random

# --------------------------------------------
# Load LUDA cache
# --------------------------------------------
cache_train = "data/cache/UCF/train/train.pkl"
cache_test  = "data/cache/UCF/test/test.pkl"

with open(cache_train, "rb") as f:
    train_data = pickle.load(f)

with open(cache_test, "rb") as f:
    test_data = pickle.load(f)

print("Train samples:", len(train_data))
print("Test samples:", len(test_data))


# --------------------------------------------
# Build Simple ResNet-based Head Detector
# --------------------------------------------
def build_detector():
    backbone = tf.keras.applications.ResNet50(
        include_top=False,
        input_shape=(256, 256, 3),
        weights="imagenet"
    )

    x = backbone.output
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)

    # 2 output heads
    heatmap = layers.Conv2D(1, 1, activation="sigmoid", name="center_heatmap")(x)
    size    = layers.Conv2D(1, 1, activation="relu",   name="pseudo_size")(x)

    return models.Model(backbone.input, [heatmap, size])


model = build_detector()
model.summary()

optimizer = optimizers.Adam(1e-4)
model.compile(
    optimizer,
    loss={"center_heatmap": "binary_crossentropy", "pseudo_size": "mse"},
    loss_weights={"center_heatmap": 1.0, "pseudo_size": 0.1}
)


# ---------------------------------------------------
#  FIXED GENERATOR → produces correct shapes
# ---------------------------------------------------
def generator(data, batch_size=1):

    while True:
        batch_imgs = []
        batch_hm = []
        batch_sz = []

        for _ in range(batch_size):
            item = random.choice(data)

            # load image
            img = cv2.imread(item["filepath"])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))
            img = img.astype(np.float32) / 255.0

            # output size = 8×8 feature map
            H, W = 8, 8
            heatmap = np.zeros((H, W, 1), np.float32)
            sizemap = np.zeros((H, W, 1), np.float32)

            # fill labels
            for (x1,y1,x2,y2) in item["bboxes"]:
                cx = int(((x1 + x2) / 2) / 256 * 8)
                cy = int(((y1 + y2) / 2) / 256 * 8)

                if 0 <= cx < 8 and 0 <= cy < 8:
                    heatmap[cy, cx, 0] = 1.0
                    sizemap[cy, cx, 0] = (y2 - y1)

            batch_imgs.append(img)
            batch_hm.append(heatmap)
            batch_sz.append(sizemap)

        yield (
            np.array(batch_imgs),
            {"center_heatmap": np.array(batch_hm),
             "pseudo_size": np.array(batch_sz)}
        )


# --------------------------------------------
# Train
# --------------------------------------------
train_gen = generator(train_data)
steps = max(1, len(train_data))

print("\n🚀 Starting FAST Training...")
model.fit(train_gen, steps_per_epoch=steps, epochs=5)


# --------------------------------------------
# Simple Evaluation – Counting
# --------------------------------------------
def evaluate():
    total_gt = 0
    total_pred = 0

    for item in test_data:
        img = cv2.imread(item["filepath"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img = img.astype(np.float32) / 255.0

        pred_hm, pred_sz = model.predict(np.expand_dims(img, 0))

        pred_count = np.sum(pred_hm[0] > 0.3)
        gt_count   = len(item["bboxes"])

        total_gt += gt_count
        total_pred += pred_count

    mae = abs(total_gt - total_pred) / len(test_data)

    print("\n📊 Evaluation Results:")
    print("GT total:", total_gt)
    print("Pred total:", total_pred)
    print("MAE:", mae)


evaluate()
os.makedirs("output_fast", exist_ok=True)
model.save("output_fast/ucf_fast_model.h5")
print("\n💾 Model saved to: output_fast/ucf_fast_model.h5")

print("\n🎉 Training + Evaluation Completed!") 