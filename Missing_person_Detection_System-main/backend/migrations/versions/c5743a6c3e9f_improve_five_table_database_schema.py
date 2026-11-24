
"""Improve five table database schema

Revision ID: c5743a6c3e9f
Revises:
Create Date: 2026-11-21 09:37:33.615209
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "c5743a6c3e9f"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():

    # --------------------------------------------------
    # 1. AUDIT LOGS
    # --------------------------------------------------

    # Backfill existing nullable columns first.
    op.execute("""
        UPDATE audit_logs
        SET "user" = 'system'
        WHERE "user" IS NULL
    """)

    op.execute("""
        UPDATE audit_logs
        SET "timestamp" = CURRENT_TIMESTAMP
        WHERE "timestamp" IS NULL
    """)

    with op.batch_alter_table("audit_logs") as batch_op:

        batch_op.add_column(
            sa.Column(
                "ip_address",
                sa.String(length=50),
                nullable=True
            )
        )

        batch_op.add_column(
            sa.Column(
                "action_status",
                sa.String(length=30),
                nullable=False,
                server_default=sa.text("'success'")
            )
        )

        batch_op.alter_column(
            "user",
            existing_type=sa.VARCHAR(length=150),
            nullable=False
        )

        batch_op.alter_column(
            "timestamp",
            existing_type=postgresql.TIMESTAMP(),
            nullable=False
        )

        batch_op.create_index(
            batch_op.f("ix_audit_logs_event_type"),
            ["event_type"],
            unique=False
        )

        batch_op.create_index(
            batch_op.f("ix_audit_logs_timestamp"),
            ["timestamp"],
            unique=False
        )

        # Remove temporary server default.
        batch_op.alter_column(
            "action_status",
            server_default=None
        )

    # --------------------------------------------------
    # 2. CAMERAS
    # --------------------------------------------------

    op.execute("""
        UPDATE cameras
        SET status = 'configured'
        WHERE status IS NULL
    """)

    op.execute("""
        UPDATE cameras
        SET created_at = CURRENT_TIMESTAMP
        WHERE created_at IS NULL
    """)

    with op.batch_alter_table("cameras") as batch_op:

        batch_op.add_column(
            sa.Column(
                "location",
                sa.String(length=255),
                nullable=True
            )
        )

        batch_op.add_column(
            sa.Column(
                "is_active",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("true")
            )
        )

        batch_op.add_column(
            sa.Column(
                "updated_at",
                sa.DateTime(),
                nullable=False,
                server_default=sa.text("CURRENT_TIMESTAMP")
            )
        )

        batch_op.alter_column(
            "status",
            existing_type=sa.VARCHAR(length=50),
            nullable=False
        )

        batch_op.alter_column(
            "created_at",
            existing_type=postgresql.TIMESTAMP(),
            nullable=False
        )

        # Remove temporary server defaults.
        batch_op.alter_column(
            "is_active",
            server_default=None
        )

        batch_op.alter_column(
            "updated_at",
            server_default=None
        )

    # --------------------------------------------------
    # 3. FACE EMBEDDINGS
    # --------------------------------------------------

    op.execute("""
        UPDATE face_embeddings
        SET model_name = 'buffalo_l'
        WHERE model_name IS NULL
    """)

    op.execute("""
        UPDATE face_embeddings
        SET created_at = CURRENT_TIMESTAMP
        WHERE created_at IS NULL
    """)

    with op.batch_alter_table("face_embeddings") as batch_op:

        batch_op.add_column(
            sa.Column(
                "embedding_dimension",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("512")
            )
        )

        batch_op.add_column(
            sa.Column(
                "is_normalized",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("true")
            )
        )

        batch_op.add_column(
            sa.Column(
                "model_version",
                sa.String(length=50),
                nullable=True
            )
        )

        batch_op.alter_column(
            "model_name",
            existing_type=sa.VARCHAR(length=100),
            nullable=False
        )

        batch_op.alter_column(
            "created_at",
            existing_type=postgresql.TIMESTAMP(),
            nullable=False
        )

        # Remove temporary server defaults.
        batch_op.alter_column(
            "embedding_dimension",
            server_default=None
        )

        batch_op.alter_column(
            "is_normalized",
            server_default=None
        )

    # --------------------------------------------------
    # 4. MATCH EVENTS
    # --------------------------------------------------

    op.execute("""
        UPDATE match_events
        SET review_status = 'pending'
        WHERE review_status IS NULL
    """)

    op.execute("""
        UPDATE match_events
        SET detected_at = CURRENT_TIMESTAMP
        WHERE detected_at IS NULL
    """)

    with op.batch_alter_table("match_events") as batch_op:

        batch_op.alter_column(
            "review_status",
            existing_type=sa.VARCHAR(length=30),
            nullable=False
        )

        batch_op.alter_column(
            "detected_at",
            existing_type=postgresql.TIMESTAMP(),
            nullable=False
        )

        batch_op.create_index(
            batch_op.f("ix_match_events_camera_id"),
            ["camera_id"],
            unique=False
        )

        batch_op.create_index(
            batch_op.f("ix_match_events_detected_at"),
            ["detected_at"],
            unique=False
        )

        batch_op.create_index(
            "ix_match_person_detected",
            ["person_id", "detected_at"],
            unique=False
        )

        # Replace the existing camera foreign key.
        batch_op.drop_constraint(
            "match_events_camera_id_fkey",
            type_="foreignkey"
        )

        batch_op.create_foreign_key(
            "fk_match_events_camera_id_cameras",
            "cameras",
            ["camera_id"],
            ["id"],
            ondelete="SET NULL"
        )

    # Add the similarity check constraint.
    op.create_check_constraint(
        "ck_match_similarity_range",
        "match_events",
        "similarity >= -1 AND similarity <= 1"
    )

    # --------------------------------------------------
    # 5. PERSONS
    # --------------------------------------------------

    op.execute("""
        UPDATE persons
        SET created_at = CURRENT_TIMESTAMP
        WHERE created_at IS NULL
    """)

    with op.batch_alter_table("persons") as batch_op:

        batch_op.add_column(
            sa.Column(
                "contact_phone",
                sa.String(length=30),
                nullable=True
            )
        )

        batch_op.add_column(
            sa.Column(
                "contact_email",
                sa.String(length=150),
                nullable=True
            )
        )

        batch_op.add_column(
            sa.Column(
                "case_status",
                sa.String(length=30),
                nullable=False,
                server_default=sa.text("'active'")
            )
        )

        batch_op.add_column(
            sa.Column(
                "updated_at",
                sa.DateTime(),
                nullable=False,
                server_default=sa.text("CURRENT_TIMESTAMP")
            )
        )

        batch_op.alter_column(
            "created_at",
            existing_type=postgresql.TIMESTAMP(),
            nullable=False
        )

        # Change name index from unique to non-unique.
        batch_op.drop_index(
            batch_op.f("ix_persons_name")
        )

        batch_op.create_index(
            batch_op.f("ix_persons_name"),
            ["name"],
            unique=False
        )

        batch_op.create_index(
            batch_op.f("ix_persons_case_status"),
            ["case_status"],
            unique=False
        )

        # Remove temporary server defaults.
        batch_op.alter_column(
            "case_status",
            server_default=None
        )

        batch_op.alter_column(
            "updated_at",
            server_default=None
        )

    # --------------------------------------------------
    # 6. FACE EMBEDDING CHECK CONSTRAINT
    # --------------------------------------------------

    op.create_check_constraint(
        "ck_embedding_dimension_positive",
        "face_embeddings",
        "embedding_dimension > 0"
    )


def downgrade():

    # --------------------------------------------------
    # 1. REMOVE CHECK CONSTRAINTS
    # --------------------------------------------------

    op.drop_constraint(
        "ck_embedding_dimension_positive",
        "face_embeddings",
        type_="check"
    )

    op.drop_constraint(
        "ck_match_similarity_range",
        "match_events",
        type_="check"
    )

    # --------------------------------------------------
    # 2. PERSONS
    # --------------------------------------------------

    with op.batch_alter_table("persons") as batch_op:

        batch_op.drop_index(
            batch_op.f("ix_persons_case_status")
        )

        batch_op.drop_index(
            batch_op.f("ix_persons_name")
        )

        # Restoring uniqueness can fail if duplicate names
        # have been inserted since the upgrade.
        batch_op.create_index(
            batch_op.f("ix_persons_name"),
            ["name"],
            unique=True
        )

        batch_op.alter_column(
            "created_at",
            existing_type=postgresql.TIMESTAMP(),
            nullable=True
        )

        batch_op.drop_column("updated_at")
        batch_op.drop_column("case_status")
        batch_op.drop_column("contact_email")
        batch_op.drop_column("contact_phone")

    # --------------------------------------------------
    # 3. MATCH EVENTS
    # --------------------------------------------------

    with op.batch_alter_table("match_events") as batch_op:

        batch_op.drop_constraint(
            "fk_match_events_camera_id_cameras",
            type_="foreignkey"
        )

        batch_op.create_foreign_key(
            "match_events_camera_id_fkey",
            "cameras",
            ["camera_id"],
            ["id"]
        )

        batch_op.drop_index("ix_match_person_detected")

        batch_op.drop_index(
            batch_op.f("ix_match_events_detected_at")
        )

        batch_op.drop_index(
            batch_op.f("ix_match_events_camera_id")
        )

        batch_op.alter_column(
            "detected_at",
            existing_type=postgresql.TIMESTAMP(),
            nullable=True
        )

        batch_op.alter_column(
            "review_status",
            existing_type=sa.VARCHAR(length=30),
            nullable=True
        )

    # --------------------------------------------------
    # 4. FACE EMBEDDINGS
    # --------------------------------------------------

    with op.batch_alter_table("face_embeddings") as batch_op:

        batch_op.alter_column(
            "created_at",
            existing_type=postgresql.TIMESTAMP(),
            nullable=True
        )

        batch_op.alter_column(
            "model_name",
            existing_type=sa.VARCHAR(length=100),
            nullable=True
        )

        batch_op.drop_column("model_version")
        batch_op.drop_column("is_normalized")
        batch_op.drop_column("embedding_dimension")

    # --------------------------------------------------
    # 5. CAMERAS
    # --------------------------------------------------

    with op.batch_alter_table("cameras") as batch_op:

        batch_op.alter_column(
            "created_at",
            existing_type=postgresql.TIMESTAMP(),
            nullable=True
        )

        batch_op.alter_column(
            "status",
            existing_type=sa.VARCHAR(length=50),
            nullable=True
        )

        batch_op.drop_column("updated_at")
        batch_op.drop_column("is_active")
        batch_op.drop_column("location")

    # --------------------------------------------------
    # 6. AUDIT LOGS
    # --------------------------------------------------

    with op.batch_alter_table("audit_logs") as batch_op:

        batch_op.drop_index(
            batch_op.f("ix_audit_logs_timestamp")
        )

        batch_op.drop_index(
            batch_op.f("ix_audit_logs_event_type")
        )

        batch_op.alter_column(
            "timestamp",
            existing_type=postgresql.TIMESTAMP(),
            nullable=True
        )

        batch_op.alter_column(
            "user",
            existing_type=sa.VARCHAR(length=150),
            nullable=True
        )

        batch_op.drop_column("action_status")
        batch_op.drop_column("ip_address")
