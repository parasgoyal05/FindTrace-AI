
import os
import re
import json
import numpy as np
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

SOURCE_DIR = Path(
    r"C:\Users\Lenovo\scikit_learn_data\lfw_home\lfw_funneled"
)

OUTPUT_DIR = BASE_DIR / "lfw_experiment"
GALLERY_DIR = OUTPUT_DIR / "gallery"
PROBE_DIR = OUTPUT_DIR / "probe"

# Use the same 158 identities selected earlier.
from sklearn.datasets import fetch_lfw_people

lfw = fetch_lfw_people(
    min_faces_per_person=10,
    color=True,
    resize=1.0
)

# Clean output folders only after you have backed them up.
import shutil

for folder in [GALLERY_DIR, PROBE_DIR]:
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)

name_mapping = {}
gallery_count = 0
probe_count = 0

for identity_id, candidate_name in enumerate(lfw.target_names):

    safe_name = re.sub(r'[<>:"/\\|?*]', "_", candidate_name)
    safe_name = safe_name.strip(" .")

    # Original LFW folder uses underscores.
    source_name = candidate_name.replace(" ", "_")
    source_person_dir = SOURCE_DIR / source_name

    if not source_person_dir.exists():
        print("[MISSING]", source_name)
        continue

    original_images = sorted(source_person_dir.glob("*.jpg"))

    if len(original_images) < 2:
        print("[SKIP]", source_name, "Not enough images")
        continue

    rng = np.random.default_rng(seed=42 + identity_id)
    shuffled = rng.permutation(len(original_images))

    gallery_person_dir = GALLERY_DIR / safe_name
    probe_person_dir = PROBE_DIR / safe_name

    gallery_person_dir.mkdir(parents=True)
    probe_person_dir.mkdir(parents=True)

    name_mapping[safe_name] = candidate_name

    # One original image for registration.
    registration_path = original_images[shuffled[0]]
    registration_image = cv2.imread(str(registration_path))

    cv2.imwrite(
        str(gallery_person_dir / "registration.jpg"),
        registration_image
    )

    gallery_count += 1

    # Remaining original images for testing.
    for number, index in enumerate(shuffled[1:], start=1):

        image_path = original_images[index]
        image = cv2.imread(str(image_path))

        cv2.imwrite(
            str(probe_person_dir / f"test_{number}.jpg"),
            image
        )

        probe_count += 1

    print(f"[DONE] {candidate_name}: {len(original_images)} images")

with open(OUTPUT_DIR / "name_mapping.json", "w", encoding="utf-8") as f:
    json.dump(name_mapping, f, indent=4)

print("\n========== ORIGINAL LFW DATASET ==========")
print("Identities:", len(name_mapping))
print("Registration images:", gallery_count)
print("Testing images:", probe_count)
print("Total images:", gallery_count + probe_count)
