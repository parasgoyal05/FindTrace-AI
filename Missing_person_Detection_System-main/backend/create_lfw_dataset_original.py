
import re
import json
import cv2
import numpy as np

from pathlib import Path
from sklearn.datasets import fetch_lfw_people

BASE_DIR = Path(__file__).resolve().parent

SOURCE_DIR = Path(
    r"C:\Users\Lenovo\scikit_learn_data\lfw_home\lfw_funneled"
)

OUTPUT_DIR = BASE_DIR / "lfw_experiment_original"
GALLERY_DIR = OUTPUT_DIR / "gallery"
PROBE_DIR = OUTPUT_DIR / "probe"

print("Loading LFW metadata...")

lfw = fetch_lfw_people(
    min_faces_per_person=10,
    color=True,
    resize=1.0
)

if not SOURCE_DIR.exists():
    raise FileNotFoundError(SOURCE_DIR)

# Never overwrite an existing experiment.
if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"{OUTPUT_DIR} already exists. Rename it before running."
    )

# First verify every identity and image count.
identity_files = {}

for identity_id, candidate_name in enumerate(lfw.target_names):

    source_name = candidate_name.replace(" ", "_")
    person_dir = SOURCE_DIR / source_name

    if not person_dir.exists():
        raise FileNotFoundError(
            f"Original identity folder missing: {person_dir}"
        )

    original_files = sorted(person_dir.glob("*.jpg"))
    expected_count = int(np.sum(lfw.target == identity_id))

    if len(original_files) != expected_count:
        raise RuntimeError(
            f"Image count mismatch for {candidate_name}: "
            f"original={len(original_files)}, "
            f"dataset={expected_count}"
        )

    identity_files[identity_id] = original_files

print("All 158 identity folders and image counts verified.")

# Create output only after verification.
GALLERY_DIR.mkdir(parents=True)
PROBE_DIR.mkdir(parents=True)

name_mapping = {}
gallery_count = 0
probe_count = 0

for identity_id, candidate_name in enumerate(lfw.target_names):

    indices = np.where(lfw.target == identity_id)[0]

    # Files are sorted in the same per-identity order.
    original_files = identity_files[identity_id]

    rng = np.random.default_rng(seed=42 + identity_id)
    shuffled_positions = rng.permutation(len(original_files))

    safe_name = re.sub(r'[<>:"/\\|?*]', "_", candidate_name)
    safe_name = safe_name.strip(" .")

    name_mapping[safe_name] = candidate_name

    gallery_person_dir = GALLERY_DIR / safe_name
    probe_person_dir = PROBE_DIR / safe_name

    gallery_person_dir.mkdir()
    probe_person_dir.mkdir()

    # One original image for registration.
    registration_file = original_files[shuffled_positions[0]]
    registration_image = cv2.imread(str(registration_file))

    if registration_image is None:
        raise RuntimeError(f"Cannot read: {registration_file}")

    cv2.imwrite(
        str(gallery_person_dir / "registration.jpg"),
        registration_image
    )

    gallery_count += 1

    # Remaining original images for testing.
    for number, position in enumerate(shuffled_positions[1:], start=1):

        image_file = original_files[position]
        image = cv2.imread(str(image_file))

        if image is None:
            raise RuntimeError(f"Cannot read: {image_file}")

        cv2.imwrite(
            str(probe_person_dir / f"test_{number}.jpg"),
            image
        )

        probe_count += 1

with open(OUTPUT_DIR / "name_mapping.json", "w", encoding="utf-8") as f:
    json.dump(name_mapping, f, indent=4)

sample_path = next(GALLERY_DIR.glob("*/registration.jpg"))
sample = cv2.imread(str(sample_path))

print("\n========== ORIGINAL LFW DATASET ==========")
print("Identities:", len(name_mapping))
print("Registration images:", gallery_count)
print("Testing images:", probe_count)
print("Total images:", gallery_count + probe_count)
print("Sample image:", sample_path)
print("Sample dimensions:", sample.shape)
print("Output folder:", OUTPUT_DIR)
