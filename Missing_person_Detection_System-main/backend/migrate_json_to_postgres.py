
import json
import math
import random
import re
from datetime import datetime, timedelta
from pathlib import Path

from app import app, db, Person, FaceEmbedding


JSON_FILE = Path(__file__).parent / "stored_embeddings.json"

# Fixed seed: the generated demo profiles remain consistent
# if the script is run again.
rng = random.Random(20261121)


# --------------------------------------------------
# SYNTHETIC US LOCATIONS
# --------------------------------------------------

US_LOCATIONS = [
    ("Grand Central Terminal, New York, NY, USA", "New York"),
    ("Penn Station, New York, NY, USA", "New York"),
    ("John F. Kennedy International Airport, NY, USA", "New York"),
    ("Los Angeles International Airport, CA, USA", "Los Angeles"),
    ("Union Station, Los Angeles, CA, USA", "Los Angeles"),
    ("Chicago Union Station, IL, USA", "Chicago"),
    ("O'Hare International Airport, IL, USA", "Chicago"),
    ("Miami International Airport, FL, USA", "Miami"),
    ("Orlando International Airport, FL, USA", "Orlando"),
    ("Dallas Union Station, TX, USA", "Dallas"),
    ("George Bush Intercontinental Airport, TX, USA", "Houston"),
    ("Seattle-Tacoma International Airport, WA, USA", "Seattle"),
    ("King Street Station, Seattle, WA, USA", "Seattle"),
    ("Boston South Station, MA, USA", "Boston"),
    ("Logan International Airport, MA, USA", "Boston"),
    ("Denver Union Station, CO, USA", "Denver"),
    ("Denver International Airport, CO, USA", "Denver"),
    ("San Francisco International Airport, CA, USA", "San Francisco"),
    ("30th Street Station, Philadelphia, PA, USA", "Philadelphia"),
    ("Ronald Reagan Washington National Airport, VA, USA", "Washington"),
    ("Church Street Marketplace, Burlington, VT, USA", "Burlington"),
    ("Nashville International Airport, TN, USA", "Nashville"),
    ("Portland Union Station, OR, USA", "Portland"),
    ("Phoenix Sky Harbor International Airport, AZ, USA", "Phoenix"),
]


# --------------------------------------------------
# SYNTHETIC DESCRIPTION GENERATOR
# --------------------------------------------------

DESCRIPTION_TEMPLATES = [
    "Fictional demo profile. Last reported near {location}. "
    "This record is generated for testing the FindTrace AI system.",

    "Synthetic missing-person case created for demonstration. "
    "Reported location: {location}. No real-world identity or incident is represented.",

    "Demo case profile for facial recognition testing. "
    "The fictional individual was last reported near {location}.",

    "Artificial test record used to evaluate database storage and "
    "face matching functionality. Reference location: {location}.",

    "Fictional person profile generated for academic demonstration. "
    "Last-seen reference: {location}."
]


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def parse_datetime(value):

    if not value:
        return datetime.utcnow()

    try:
        parsed = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )

        if parsed.tzinfo is not None:
            parsed = parsed.replace(tzinfo=None)

        return parsed

    except (ValueError, TypeError, AttributeError):
        return datetime.utcnow()


def generate_demo_profile(index):

    age = rng.randint(31, 85)
    gender = rng.choice(["Male", "Female"])

    location, city = rng.choice(US_LOCATIONS)

    days_ago = rng.randint(1, 730)

    last_seen_date = (
        datetime.utcnow()
        - timedelta(days=days_ago)
    )

    description = rng.choice(
        DESCRIPTION_TEMPLATES
    ).format(location=location)

    return {
        "age": age,
        "gender": gender,
        "last_seen_location": location,
        "last_seen_date": last_seen_date,
        "contact_info": "Synthetic demo contact - not real",
        "contact_phone": "000-000-0000",
        "contact_email": f"demo-person-{index}@example.com",
        "description": description,
        "case_status": "active",
    }


def validate_embedding(name, record):

    if not isinstance(record, dict):
        raise ValueError(f"Invalid JSON record: {name}")

    embedding = record.get("embedding")

    if not isinstance(embedding, list):
        raise ValueError(f"Embedding missing for: {name}")

    embedding = [
        float(value) for value in embedding
    ]

    if len(embedding) != 512:
        raise ValueError(
            f"{name}: Expected 512 values, got {len(embedding)}"
        )

    if not all(math.isfinite(value) for value in embedding):
        raise ValueError(
            f"{name}: Embedding contains invalid numbers"
        )

    return embedding


# --------------------------------------------------
# MAIN MIGRATION
# --------------------------------------------------

def migrate_embeddings():

    if not JSON_FILE.exists():
        raise FileNotFoundError(
            f"JSON file not found: {JSON_FILE}"
        )

    with open(JSON_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "Expected JSON to contain a dictionary of persons."
        )

    print("=" * 55)
    print("FINDTRACE AI - JSON TO POSTGRESQL MIGRATION")
    print("=" * 55)

    print(f"Persons found in JSON: {len(data)}")

    inserted_persons = 0
    updated_persons = 0
    inserted_embeddings = 0
    updated_embeddings = 0

    with app.app_context():

        try:

            for index, (name, record) in enumerate(
                data.items(), start=1
            ):

                if not isinstance(name, str) or not name.strip():
                    raise ValueError(
                        f"Invalid person name at record {index}"
                    )

                embedding = validate_embedding(
                    name, record
                )

                created_at = parse_datetime(
                    record.get("created_at")
                )

                # Generate complete synthetic profile.
                profile = generate_demo_profile(index)

                # Find existing person to make this script rerunnable.
                person = Person.query.filter_by(
                    name=name
                ).first()

                if person is None:

                    person = Person(name=name)

                    db.session.add(person)

                    inserted_persons += 1

                else:

                    updated_persons += 1

                # Populate every person field.
                person.age = profile["age"]
                person.gender = profile["gender"]

                person.last_seen_location = (
                    profile["last_seen_location"]
                )

                person.last_seen_date = (
                    profile["last_seen_date"]
                )

                person.contact_info = (
                    profile["contact_info"]
                )

                person.contact_phone = (
                    profile["contact_phone"]
                )

                person.contact_email = (
                    profile["contact_email"]
                )

                person.description = (
                    profile["description"]
                )

                person.case_status = "active"

                person.created_at = created_at
                person.updated_at = datetime.utcnow()

                person.image_filename = (
                    record.get("image_filename")
                    or f"synthetic_demo_image_{index}.jpg"
                )

                person.uploader_ip = (
                    record.get("uploader_ip")
                    or "127.0.0.1"
                )

                person.source = "synthetic_demo_data"

                # Generate the person ID.
                db.session.flush()

                # Calculate whether embedding is normalized.
                norm = math.sqrt(
                    sum(value * value for value in embedding)
                )

                is_normalized = (
                    abs(norm - 1.0) < 0.001
                )

                # Check whether an embedding already exists.
                face_embedding = (
                    FaceEmbedding.query
                    .filter_by(person_id=person.id)
                    .order_by(FaceEmbedding.id.asc())
                    .first()
                )

                if face_embedding is None:

                    face_embedding = FaceEmbedding(
                        person_id=person.id
                    )

                    db.session.add(face_embedding)

                    inserted_embeddings += 1

                else:

                    updated_embeddings += 1

                # Preserve the original embedding values.
                face_embedding.embedding = embedding

                face_embedding.embedding_dimension = 512
                face_embedding.is_normalized = is_normalized
                face_embedding.model_name = "buffalo_l"
                face_embedding.model_version = "buffalo_l"
                face_embedding.image_filename = (
                    person.image_filename
                )
                face_embedding.created_at = created_at

                print(
                    f"[{index}/{len(data)}] "
                    f"Migrated: {name} | "
                    f"Age: {profile['age']} | "
                    f"Gender: {profile['gender']}"
                )

            # Save everything in one transaction.
            db.session.commit()

            print("\n" + "=" * 55)
            print("MIGRATION COMPLETED SUCCESSFULLY")
            print("=" * 55)

            print(f"New persons inserted: {inserted_persons}")
            print(f"Existing persons updated: {updated_persons}")

            print(
                f"New embeddings inserted: {inserted_embeddings}"
            )

            print(
                f"Existing embeddings updated: {updated_embeddings}"
            )

            print("\nPOSTGRESQL DATABASE COUNTS")

            print(
                "Total persons:",
                Person.query.count()
            )

            print(
                "Total embeddings:",
                FaceEmbedding.query.count()
            )

            print("\nMigration verification completed.")

        except Exception:

            db.session.rollback()

            print("\nMIGRATION FAILED")
            print("Database transaction rolled back.")

            raise

        finally:

            db.session.remove()


if __name__ == "__main__":
    migrate_embeddings()
