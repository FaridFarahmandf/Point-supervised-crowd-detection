import os
import cv2
import numpy as np
import scipy.spatial
import scipy.io
import pickle


# =========================================
# Process one split (train or test)
# =========================================
def process_split(split):

    print("\n=====================================")
    print("Processing UCF Mini Split:", split)
    print("=====================================\n")

    root_dir = "./data/UCF"
    img_path = os.path.join(root_dir, "TrainMini" if split == "train" else "TestMini")

    # output directory
    cache_root = "data/cache/UCF"
    out_dir = os.path.join(cache_root, split)
    os.makedirs(out_dir, exist_ok=True)

    img_names = [n for n in os.listdir(img_path) if n.endswith(".jpg")]
    print("Found", len(img_names), "images.")

    image_data = []
    weights_list = []
    scales = []

    total_imgs = 0
    valid_imgs = 0
    total_points = 0

    for img_name in img_names:
        total_imgs += 1
        full_img_path = os.path.join(img_path, img_name)

        img = cv2.imread(full_img_path)
        if img is None:
            print("Warning: can't read:", full_img_path)
            continue

        H, W = img.shape[:2]

        # annotation file
        mat_name = img_name.replace(".jpg", "_ann.mat")
        mat_path = os.path.join(img_path, mat_name)

        if not os.path.exists(mat_path):
            print("Missing .mat annotation:", mat_path)
            continue

        data = scipy.io.loadmat(mat_path)

        if "annPoints" not in data:
            print("Invalid annotation:", mat_path)
            continue

        centers = data["annPoints"].astype(np.float32)
        N = centers.shape[0]
        total_points += N

        if N == 0:
            continue

        valid_imgs += 1

        # ----------------------------------------
        # LUDA pseudo-size computation
        # ----------------------------------------
        if N <= 2:
            scale = np.ones(N) * max(H / 6.0, 12)
            scale_weight = np.ones(N)
        else:
            tree = scipy.spatial.KDTree(centers.copy())
            k = min(3, N)
            dist, idx = tree.query(centers, k=min(N, max(3, N//2)))

            mean_dist = np.mean(dist[:, 1:k], axis=1)
            crowd_range = np.max(centers[:, 1]) - np.min(centers[:, 1])
            circle_scale = crowd_range / 10.0

            in_circle = np.where(dist <= circle_scale)
            ids, counts = np.unique(in_circle[0], return_counts=True)

            scale = np.zeros(N)
            scale_weight = np.zeros(N)

            for u, c in zip(ids, counts):
                local_idx = idx[u, :c]
                s = np.mean(mean_dist[local_idx])
                s = np.clip(s, 2, None)
                scale[u] = s
                scale_weight[u] = c

            for i in range(N):
                if scale[i] == 0:
                    scale[i] = np.clip(mean_dist[i], 2, None)
                    scale_weight[i] = 1.0

        # ----------------------------------------
        # Form bounding boxes (pseudo sizes)
        # ----------------------------------------
        boxes = np.zeros((N, 4), dtype=np.float32)
        boxes[:, 0] = centers[:, 0] - scale / 2
        boxes[:, 1] = centers[:, 1] - scale / 2
        boxes[:, 2] = centers[:, 0] + scale / 2
        boxes[:, 3] = centers[:, 1] + scale / 2

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, W - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, H - 1)

        annotation = {
            "filepath": full_img_path,
            "bboxes": boxes,
            "confs": 0.6 * np.ones(N),
            "w_bboxes": scale_weight,
            "ignoreareas": np.zeros((0, 4), np.float32),
        }

        image_data.append(annotation)
        weights_list.extend(scale_weight)
        scales.extend(scale)

    # ----------------------------------------
    # Stats + save output
    # ----------------------------------------
    weights = np.array(weights_list)
    scales = np.array(scales)

    print(f"Weights max: {weights.max():.2f}")
    print(f"Weights mean: {weights.mean():.2f}")
    print(f"Weights std: {weights.std():.2f}")

    print(f"Scales max: {scales.max():.2f}")
    print(f"Scales min: {scales.min():.2f}")
    print(f"Scales mean: {scales.mean():.2f}")
    print(f"Scales std: {scales.std():.2f}")

    print(f"{total_imgs} images total, {valid_imgs} valid, {total_points} total points")

    out_file = os.path.join(out_dir, f"{split}.pkl")
    with open(out_file, "wb") as f:
        pickle.dump(image_data, f, pickle.HIGHEST_PROTOCOL)

    print("\nSaved:", out_file)
    print("=====================================\n")


# =====================================================
# Run for both train and test
# =====================================================
process_split("train")
process_split("test")

print("🎉 All done! Train + Test LUDA files generated.")
