"""
Flask Backend API for Missing Person Detection
Handles face embedding computation and storage using DeepFace
"""
from dotenv import load_dotenv
load_dotenv()
import os
import sys
import json
from flask import Flask, request, jsonify, render_template, Response, session
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import threading
import base64
import cv2
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
import time
import secrets
from functools import wraps

# Fix Windows encoding issues
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


class AlertSystem:
    """Enhanced alert system for sending notifications when missing persons are detected"""
    
    def __init__(self):
        self.alerts_sent = set()  # Track sent alerts to avoid spam (time bucket key)
        self.last_alert_times = {}  # Precise per-person/camera timestamp for cooldown
        self.alert_cooldown = 30  # 30 seconds cooldown between alerts for same person/camera
        self.alert_threshold = 0.55  # Minimum similarity to trigger audible alert (strong matches)
        self.alerts_history = []  # Store alert history
        self.snapshots_folder = 'snapshots'
        self.alerts_folder = 'alerts'
        
        # Create necessary directories
        os.makedirs(self.snapshots_folder, exist_ok=True)
        os.makedirs(self.alerts_folder, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.alerts_folder, 'alerts.log')),
                logging.StreamHandler()
            ]
        )
        logging.getLogger().setLevel(logging.WARNING)
    
    def save_snapshot(self, frame, match_data, camera_source=None):
        """
        Save snapshot of detected person
        
        Args:
            frame: numpy array of the frame
            match_data: dict with match information
            camera_source: source identifier for the camera
        """
        try:
            timestamp = datetime.now()
            person_name = match_data['name']
            confidence = match_data.get('similarity', 0)
            
            # Create filename with timestamp and person name
            safe_name = "".join(c for c in person_name if c.isalnum() or c in (' ', '-', '_')).strip()
            filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{safe_name}_{confidence:.2f}.jpg"
            if camera_source:
                filename = f"{camera_source}_{filename}"
            
            filepath = os.path.join(self.snapshots_folder, filename)
            
            # Draw bounding box and label on snapshot if available
            annotated_frame = frame.copy()
            if 'bbox' in match_data:
                bbox = match_data['bbox']
                x, y, w, h = bbox
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                label = f"{person_name} ({confidence*100:.1f}%)"
                cv2.putText(annotated_frame, label, (x, y - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Save snapshot
            cv2.imwrite(filepath, annotated_frame)
            return filepath
        except Exception as e:
            logging.error(f"Error saving snapshot: {e}")
            return None
    
    def send_alert(self, match_data, frame=None, frame_info=None, camera_source=None):
        """
        Send comprehensive alert for detected missing person with snapshot
        
        Args:
            match_data: dict with match information
            frame: numpy array of the frame (for snapshot)
            frame_info: optional frame information
            camera_source: source identifier for the camera
        """
        person_name = match_data['name']
        confidence = match_data.get('similarity', match_data.get('confidence', '0%'))
        if isinstance(confidence, str):
            confidence_val = float(confidence.replace('%', '')) / 100
        else:
            confidence_val = float(confidence)
        
        timestamp = datetime.now().isoformat()
        
# Enforce strong-match threshold before alerting
        if confidence_val < self.alert_threshold:
            logging.debug(f"[ALERT SKIP] {person_name}: similarity {confidence_val:.2f} < threshold {self.alert_threshold:.2f}")
            return

        # Check if we recently sent an alert for this person+camera (cooldown)
        base_key = f"{person_name}_{camera_source}"
        current_time = datetime.now().timestamp()

        last_time = self.last_alert_times.get(base_key, 0)
        if current_time - last_time < self.alert_cooldown:
            logging.debug(f"[ALERT SKIP] {person_name}: cooldown active ({current_time-last_time:.1f}s < {self.alert_cooldown}s)")
            return

        self.last_alert_times[base_key] = current_time

        alert_bucket_key = f"{person_name}_{camera_source}_{int(current_time // self.alert_cooldown)}"
        self.alerts_sent.add(alert_bucket_key)

        # Clean old alerts (keep only recent bucket keys within last hour)
        cutoff = current_time - 3600
        self.alerts_sent = {k for k in self.alerts_sent 
                            if int(k.split('_')[-1]) * self.alert_cooldown > cutoff}
        self.last_alert_times = {k: v for k, v in self.last_alert_times.items() if v > cutoff}

        logging.debug(f"[ALERT TRIGGER] {person_name} ({confidence_val*100:.2f}%) - base_key: {base_key}")
        
        # Save snapshot if frame is provided
        snapshot_path = None
        if frame is not None:
            snapshot_path = self.save_snapshot(frame, match_data, camera_source)
        
        # Create alert record
        alert_record = {
            'person_name': person_name,
            'confidence': confidence_val,
            'confidence_str': f"{confidence_val * 100:.2f}%",
            'timestamp': timestamp,
            'camera_source': camera_source or 'unknown',
            'snapshot_path': snapshot_path,
            'frame_info': frame_info or {}
        }
        
        # Add to history
        self.alerts_history.append(alert_record)
        if len(self.alerts_history) > 1000:  # Keep last 1000 alerts
            self.alerts_history = self.alerts_history[-1000:]
        
        # Log alert
        logging.warning(f"ALERT: Missing person '{person_name}' detected with {confidence_val*100:.2f}% confidence at {timestamp} (Camera: {camera_source})")
        
        # Send system notification
        self._send_system_alert(alert_record)
        
        # Emit alert via WebSocket with snapshot
        alert_data = {
            'person_name': person_name,
            'confidence': f"{confidence_val * 100:.2f}%",
            'confidence_value': confidence_val,
            'timestamp': timestamp,
            'camera_source': camera_source or 'unknown',
            'message': f"🚨 ALERT: {person_name} detected!",
            'snapshot_path': snapshot_path,
            'snapshot_url': f'/api/snapshots/{os.path.basename(snapshot_path)}' if snapshot_path else None
        }
        
        try:
            logging.debug(f"[SOCKETIO] Emitting alert_triggered for {person_name}")
            socketio.emit('alert_triggered', alert_data, broadcast=True)
            logging.debug("[SOCKETIO] Alert emitted successfully")
        except Exception as e:
            logging.error(f"[SOCKETIO ERROR] Failed to emit alert: {e}")
        
        return alert_record
    
    def _send_system_alert(self, alert_record):
        """Send system alert with comprehensive information"""
        try:
            alert_message = f"""
{'='*60}
🚨 MISSING PERSON ALERT 🚨
{'='*60}

Person Detected: {alert_record['person_name']}
Confidence: {alert_record['confidence_str']}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Camera Source: {alert_record['camera_source']}
Snapshot: {alert_record['snapshot_path'] or 'Not available'}

Please check the live feed immediately!
{'='*60}
"""
            print(alert_message)
            
            # TODO: Integrate with email service (SendGrid, SMTP) or SMS (Twilio)
            # Example: self._send_email_alert(alert_record)
            # Example: self._send_sms_alert(alert_record)
            
        except Exception as e:
            logging.error(f"Error sending system alert: {e}")
    
    def get_alert_history(self, limit=50, person_name=None):
        """Get alert history with optional filtering"""
        history = self.alerts_history.copy()
        
        if person_name:
            history = [a for a in history if a['person_name'].lower() == person_name.lower()]
        
        return history[-limit:] if history else []
    
    def get_snapshot(self, filename):
        """Get snapshot file path"""
        filepath = os.path.join(self.snapshots_folder, filename)
        if os.path.exists(filepath):
            return filepath
        return None


# Initialize alert system
alert_system = AlertSystem()

# Match dedupe and throttling across frames
match_emit_cooldown = 10  # seconds cooldown per person + camera
match_last_emitted = {}

# Initialize Flask app
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(32))
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
CORS(app, supports_credentials=True)

# ================= POSTGRESQL CONFIGURATION =================

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. Configure it as an environment variable "
        "before starting FindTrace (do not hardcode database credentials in app.py)."
    )
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)


# ================= DATABASE MODELS =================

# ================= DATABASE MODELS =================

class Person(db.Model):
    __tablename__ = "persons"

    id = db.Column(db.Integer, primary_key=True)

    # Identity
    name = db.Column(db.String(150), nullable=False, index=True)
    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(30), nullable=True)

    # Missing person information
    last_seen_location = db.Column(db.String(255), nullable=True)
    last_seen_date = db.Column(db.DateTime, nullable=True)
    contact_info = db.Column(db.String(100), nullable=True)
    contact_phone = db.Column(db.String(30), nullable=True)
    contact_email = db.Column(db.String(150), nullable=True)
    description = db.Column(db.Text, nullable=True)

    # Case management
    case_status = db.Column(
        db.String(30),
        nullable=False,
        default="active",
        index=True
    )

    # Registration metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )
    image_filename = db.Column(db.String(255), nullable=True)
    uploader_ip = db.Column(db.String(50), nullable=True)
    source = db.Column(db.String(100), nullable=True)

    # Relationships
    embeddings = db.relationship(
        "FaceEmbedding",
        backref="person",
        cascade="all, delete-orphan",
        lazy=True
    )

    match_events = db.relationship(
        "MatchEvent",
        backref="person",
        lazy=True
    )


class FaceEmbedding(db.Model):
    __tablename__ = "face_embeddings"

    id = db.Column(db.Integer, primary_key=True)

    person_id = db.Column(
        db.Integer,
        db.ForeignKey("persons.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # ArcFace embedding
    embedding = db.Column(db.JSON, nullable=False)
    embedding_dimension = db.Column(
        db.Integer,
        nullable=False,
        default=512
    )
    is_normalized = db.Column(
        db.Boolean,
        nullable=False,
        default=True
    )

    # Model information
    model_name = db.Column(
        db.String(100),
        nullable=False,
        default="buffalo_l"
    )
    model_version = db.Column(db.String(50), nullable=True)

    image_filename = db.Column(db.String(255), nullable=True)
    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    __table_args__ = (
        db.CheckConstraint(
            "embedding_dimension > 0",
            name="ck_embedding_dimension_positive"
        ),
    )


class Camera(db.Model):
    __tablename__ = "cameras"

    id = db.Column(db.Integer, primary_key=True)

    camera_name = db.Column(db.String(150), nullable=False)
    camera_type = db.Column(db.String(50), nullable=False)

    # Webcam index, RTSP URL, IP camera URL, etc.
    source = db.Column(db.Text, nullable=False, unique=True)

    location = db.Column(db.String(255), nullable=True)

    status = db.Column(
        db.String(50),
        nullable=False,
        default="configured"
    )

    is_active = db.Column(
        db.Boolean,
        nullable=False,
        default=True
    )

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )


class MatchEvent(db.Model):
    __tablename__ = "match_events"

    id = db.Column(db.Integer, primary_key=True)

    person_id = db.Column(
        db.Integer,
        db.ForeignKey("persons.id"),
        nullable=False,
        index=True
    )

    camera_id = db.Column(
        db.Integer,
        db.ForeignKey("cameras.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )

    # Cosine similarity score
    similarity = db.Column(db.Float, nullable=False)

    camera_source = db.Column(db.String(255), nullable=True)
    snapshot_path = db.Column(db.String(500), nullable=True)

    review_status = db.Column(
        db.String(30),
        nullable=False,
        default="pending"
    )

    detected_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True
    )

    camera = db.relationship("Camera", backref="match_events")

    __table_args__ = (
        db.CheckConstraint(
            "similarity >= -1 AND similarity <= 1",
            name="ck_match_similarity_range"
        ),
        db.Index(
            "ix_match_person_detected",
            "person_id",
            "detected_at"
        ),
    )


class AuditLog(db.Model):
    __tablename__ = "audit_logs"

    id = db.Column(db.Integer, primary_key=True)

    event_type = db.Column(db.String(100), nullable=False, index=True)
    description = db.Column(db.Text, nullable=False)

    user = db.Column(
        db.String(150),
        nullable=False,
        default="system"
    )

    ip_address = db.Column(db.String(50), nullable=True)

    action_status = db.Column(
        db.String(30),
        nullable=False,
        default="success"
    )

    details = db.Column(db.JSON, nullable=True)

    timestamp = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True
    )

# ================= PUBLIC CASES & ADMIN AUTH =================

class Admin(db.Model):
    __tablename__ = "admins"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


class CaseSubmission(db.Model):
    __tablename__ = "case_submissions"

    id = db.Column(db.Integer, primary_key=True)
    case_id = db.Column(db.String(24), unique=True, nullable=False, index=True)

    # Person details supplied by the public
    person_name = db.Column(db.String(150), nullable=False)
    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(30), nullable=True)
    last_seen_location = db.Column(db.String(255), nullable=False)
    last_seen_date = db.Column(db.DateTime, nullable=True)
    description = db.Column(db.Text, nullable=True)
    image_filename = db.Column(db.String(255), nullable=False)

    # Reporter details are private and are never returned by public tracking
    reporter_name = db.Column(db.String(150), nullable=False)
    reporter_phone = db.Column(db.String(30), nullable=False)
    reporter_email = db.Column(db.String(150), nullable=True)

    status = db.Column(db.String(20), nullable=False, default="pending", index=True)
    review_note = db.Column(db.Text, nullable=True)
    reviewed_by = db.Column(db.String(80), nullable=True)
    reviewed_at = db.Column(db.DateTime, nullable=True)
    person_id = db.Column(db.Integer, db.ForeignKey("persons.id"), nullable=True)
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


def admin_required(fn):
    """Protect JSON/API endpoints with the authenticated admin session."""
    @wraps(fn)
    def decorated(*args, **kwargs):
        if not session.get("admin_id"):
            return jsonify({"success": False, "error": "Admin login required"}), 401
        return fn(*args, **kwargs)
    return decorated


def admin_page_required(fn):
    """Send unauthenticated visitors to the admin sign-in page."""
    @wraps(fn)
    def decorated(*args, **kwargs):
        if not session.get("admin_id"):
            from flask import redirect, url_for
            return redirect(url_for("admin_page"))
        return fn(*args, **kwargs)
    return decorated


def create_case_id():
    # Human-readable prefix plus a cryptographically random suffix.
    return "FT-" + secrets.token_hex(5).upper()


def case_submission_payload(case, include_private=False):
    payload = {
        "id": case.id,
        "case_id": case.case_id,
        "person_name": case.person_name,
        "age": case.age,
        "gender": case.gender,
        "last_seen_location": case.last_seen_location,
        "last_seen_date": case.last_seen_date.isoformat() if case.last_seen_date else None,
        "description": case.description,
        "image_url": f"/uploads/{case.image_filename}",
        "status": case.status,
        "review_note": case.review_note,
        "submitted_at": case.submitted_at.isoformat() if case.submitted_at else None,
        "reviewed_at": case.reviewed_at.isoformat() if case.reviewed_at else None,
        "person_id": case.person_id,
    }
    if include_private:
        payload.update({
            "reporter_name": case.reporter_name,
            "reporter_phone": case.reporter_phone,
            "reporter_email": case.reporter_email,
            "reviewed_by": case.reviewed_by,
        })
    return payload


print("PostgreSQL configuration loaded")

def get_or_create_camera_record(name, source, camera_type=None, location=None):
    """Create or update a camera record in PostgreSQL, keyed by source."""
    source_key = str(source)
    inferred_type = camera_type or (
        "webcam" if source_key.isdigit()
        else "ip_camera" if source_key.lower().startswith(("rtsp://", "rtsps://", "http://", "https://"))
        else "camera"
    )
    camera = Camera.query.filter_by(source=source_key).first()
    if camera is None:
        camera = Camera(
            camera_name=(name or source_key)[:150],
            camera_type=inferred_type[:50],
            source=source_key,
            location=location,
            status="configured",
            is_active=True
        )
        db.session.add(camera)
    else:
        camera.camera_name = (name or camera.camera_name)[:150]
        camera.camera_type = inferred_type[:50]
        if location:
            camera.location = location
        camera.is_active = True
    db.session.flush()
    return camera


def set_camera_status(source, status, is_active=None):
    """Update a configured camera's operational status in PostgreSQL."""
    try:
        with app.app_context():
            camera = Camera.query.filter_by(source=str(source)).first()
            if camera:
                camera.status = status
                if is_active is not None:
                    camera.is_active = is_active
                db.session.commit()
    except Exception:
        try:
            with app.app_context():
                db.session.rollback()
        except Exception:
            pass
        logging.exception("Could not update camera status in PostgreSQL")


def save_match_event(match, camera_source, camera_id=None, snapshot_path=None):
    """Persist a deduplicated face match in PostgreSQL."""
    try:
        with app.app_context():
            person_id = match.get("person_id")
            person = None
            if person_id is not None:
                try:
                    person = db.session.get(Person, int(person_id))
                except (TypeError, ValueError):
                    person = None
            if person is None:
                person = Person.query.filter_by(name=match.get("name")).first()
            if person is None:
                logging.warning("Skipping match event: person %r not found", match.get("name"))
                return False

            camera = db.session.get(Camera, camera_id) if camera_id else None
            if camera is None:
                camera = Camera.query.filter_by(source=str(camera_source)).first()

            raw_similarity = match.get("similarity", match.get("confidence", 0))
            if isinstance(raw_similarity, str):
                raw_similarity = raw_similarity.strip()
                raw_similarity = float(raw_similarity.replace("%", "")) / (100 if "%" in raw_similarity else 1)
            similarity = max(-1.0, min(1.0, float(raw_similarity)))

            db.session.add(MatchEvent(
                person_id=person.id,
                camera_id=camera.id if camera else None,
                similarity=similarity,
                camera_source=str(camera_source)[:255] if camera_source is not None else None,
                snapshot_path=snapshot_path,
                review_status="pending",
                detected_at=datetime.utcnow()
            ))
            db.session.commit()
            return True
    except Exception:
        db.session.rollback()
        logging.exception("Could not save match event in PostgreSQL")
        return False





@app.route('/create-tables')
def create_tables():
    try:
        db.create_all()

        return jsonify({
            "success": True,
            "message": "All FindTrace database tables created!"
        }), 200

    except Exception as e:
        logging.exception("Database table creation failed")

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/test-db')
def test_db():
    try:
        with db.engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        return jsonify({
            "success": True,
            "message": "PostgreSQL connected successfully!"
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500



socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    logger=False,
    engineio_logger=False
)
# ================== ArcFace Registration Model ==================
# This model will be used when user uploads a missing person photo

from insightface.app import FaceAnalysis


def _gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


gpu_available = _gpu_available()
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu_available else ["CPUExecutionProvider"]
print("Loading ArcFace model for registration (database embeddings)...")
print(f"ArcFace registration GPU available: {gpu_available} | providers: {providers}")

arcface_model = FaceAnalysis(
    name="buffalo_l",
    providers=providers
)

arcface_model.prepare(ctx_id=0 if gpu_available else -1, det_size=(640, 640))

print("ArcFace registration model loaded successfully")


# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
EMBEDDINGS_FILE = 'stored_embeddings.json'
EVIDENCE_FOLDER = 'evidence'
AUDIT_LOG_FILE = 'audit.log'
# Path to configured CCTV/Drone streams (project root / streams / cameras.json)
STREAMS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'streams'))
CAMERAS_CONFIG_FILE = os.path.join(STREAMS_DIR, 'cameras.json')

# Create evidence directory
os.makedirs(EVIDENCE_FOLDER, exist_ok=True)

# Import real-time detector (after configuration)
from realtime_detector import RealTimeDetector, VideoStreamProcessor

# Initialize real-time detector (use absolute path to trained model `best.pt` in backend dir)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best.pt')
print(f"Using YOLO model path: {MODEL_PATH}")
realtime_detector = RealTimeDetector(embeddings_file=EMBEDDINGS_FILE, threshold=0.30, frame_skip=1, model_path=MODEL_PATH)
stream_processor = VideoStreamProcessor(realtime_detector)
processing_thread = None


def load_postgres_embeddings():
    """Load the latest face embeddings from PostgreSQL into the detector."""
    try:
        records = (
            db.session.query(Person, FaceEmbedding)
            .join(FaceEmbedding, FaceEmbedding.person_id == Person.id)
            .filter(Person.case_status == "active")
            .order_by(FaceEmbedding.created_at.desc())
            .all()
        )

        embeddings_dict = {}
        for person, face_record in records:
            vector = face_record.embedding
            if not isinstance(vector, (list, tuple)) or len(vector) != 512:
                logging.warning(
                    "Skipping invalid embedding for person %s (id=%s)",
                    person.name, person.id
                )
                continue

            # Keep the newest embedding for each person name.
            embeddings_dict.setdefault(person.name, {
                "embedding": vector,
                "created_at": (
                    face_record.created_at.isoformat()
                    if face_record.created_at else None
                ),
                "image_filename": face_record.image_filename or person.image_filename,
                "person_id": person.id
            })

        realtime_detector.load_embeddings_from_dict(embeddings_dict)
        print(
            f"[PostgreSQL] Loaded {len(realtime_detector.stored_embeddings_np)} "
            "face embeddings into the detector"
        )
        return len(realtime_detector.stored_embeddings_np)

    except Exception:
        db.session.rollback()
        logging.exception("Could not load face embeddings from PostgreSQL")
        raise


# Load database embeddings after the detector and SQLAlchemy models are ready.
with app.app_context():
    load_postgres_embeddings()


# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set maximum file size (16MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/uploads/<path:filename>', methods=['GET'])
@admin_required
def serve_uploaded_image(filename):
    """Serve registered missing-person photographs."""
    from flask import send_from_directory

    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        filename
    )


def stream_callback(result, frame):

    _, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

    socketio.emit('frame_update', {
        'frame': jpg_as_text
    })

    if result["matches"]:
        socketio.emit("match_found", {
            "matches": result["matches"],
            "timestamp": datetime.now().isoformat()
        })
        
def start_stream_thread(source, camera_name):

    global processing_thread

    def run():
        stream_processor.open_camera(source, camera_name)
        stream_processor.process_stream(callback=stream_callback)

    processing_thread = threading.Thread(target=run)
    processing_thread.daemon = True
    processing_thread.start()
        



def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_embeddings():
    """Load stored embeddings from JSON file"""
    if os.path.exists(EMBEDDINGS_FILE):
        try:
            with open(EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError, UnicodeDecodeError) as e:
            print(f"Error loading embeddings: {e}")
            return {}
    return {}


@app.route('/preview-json-migration', methods=['GET'])
@admin_required
def preview_json_migration():
    try:
        if not os.path.exists(EMBEDDINGS_FILE):
            return jsonify({
                "success": False,
                "error": "stored_embeddings.json not found"
            }), 404

        embeddings_dict = load_embeddings()

        if not isinstance(embeddings_dict, dict):
            return jsonify({
                "success": False,
                "error": "Invalid JSON structure"
            }), 400

        existing_names = {
            name for (name,) in
            db.session.query(Person.name).all()
        }

        valid = 0
        invalid = 0
        already_exists = 0

        for name, data in embeddings_dict.items():
            if not isinstance(data, dict):
                invalid += 1
                continue

            embedding = data.get("embedding")

            try:
                vector = np.asarray(embedding, dtype=np.float32)

                if vector.shape != (512,) or not np.isfinite(vector).all():
                    invalid += 1
                    continue

            except (TypeError, ValueError):
                invalid += 1
                continue

            if not isinstance(name, str) or not name.strip():
                invalid += 1
            elif name in existing_names:
                already_exists += 1
            else:
                valid += 1

        return jsonify({
            "success": True,
            "total_json_records": len(embeddings_dict),
            "ready_to_migrate": valid,
            "invalid_records": invalid,
            "already_in_postgres": already_exists,
            "database_connection": "successful"
        }), 200

    except Exception as e:
        logging.exception("Migration preview failed")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


def save_embeddings(embeddings_dict):
    """Save embeddings dictionary to JSON file"""
    try:
        with open(EMBEDDINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(embeddings_dict, f, indent=2, ensure_ascii=False, default=str)
        return True
    except (IOError, UnicodeEncodeError) as e:
        error_msg = str(e).encode('ascii', 'replace').decode('ascii')
        print(f"Error saving embeddings: {error_msg}")
        return False

def compute_face_embedding(image_path):
    """
    Generate ArcFace embedding for missing person registration
    """

    try:
        img = cv2.imread(image_path)

        if img is None:
            print("Image could not be read")
            return None

        # Detect face and generate embedding
        faces = arcface_model.get(img)

        if len(faces) == 0:
            print("No face detected in uploaded image")
            return None

        embedding = faces[0].embedding

        # Normalize embedding (CRITICAL for cosine similarity)
        embedding = embedding / np.linalg.norm(embedding)

        return embedding.tolist()

    except Exception as e:
        print("ArcFace registration error:", e)
        return None


def find_matching_person(detected_embedding, threshold=0.6):
    """Compare a supplied face embedding against embeddings loaded from PostgreSQL."""
    matches = []
    if not getattr(realtime_detector, "stored_embeddings_np", None):
        return matches

    detected = np.asarray(detected_embedding, dtype=np.float32).reshape(1, -1)
    if detected.shape[1] != 512 or not np.isfinite(detected).all():
        return matches

    for name, stored_embedding in realtime_detector.stored_embeddings_np.items():
        stored = np.asarray(stored_embedding, dtype=np.float32).reshape(1, -1)
        if stored.shape != detected.shape or not np.isfinite(stored).all():
            continue

        similarity = float(cosine_similarity(detected, stored)[0][0])
        if similarity >= threshold:
            metadata = realtime_detector.stored_embeddings.get(name, {})
            matches.append({
                "name": name,
                "similarity": similarity,
                "confidence": f"{similarity * 100:.2f}%",
                "created_at": metadata.get("created_at")
            })

    matches.sort(key=lambda item: item["similarity"], reverse=True)
    return matches




@app.route('/target_person', methods=['POST'])
@admin_required
def add_target_person():
    """Register a missing person and their 512-dimensional face embedding in PostgreSQL."""
    filepath = None

    if 'name' not in request.form:
        return jsonify({'success': False, 'error': 'Missing required field: name'}), 400

    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'Missing required field: image'}), 400

    name = request.form.get('name', '').strip()
    image_file = request.files['image']

    if not name:
        return jsonify({'success': False, 'error': 'Name cannot be empty'}), 400

    if image_file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400

    if not allowed_file(image_file.filename):
        return jsonify({
            'success': False,
            'error': f'Invalid file type. Allowed types: {", ".join(sorted(ALLOWED_EXTENSIONS))}'
        }), 400

    # The existing endpoint treats a repeated name as a conflict.
    if Person.query.filter_by(name=name).first():
        return jsonify({
            'success': False,
            'error': f'Person with name "{name}" already exists.'
        }), 409

    filename = secure_filename(image_file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    unique_filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

    try:
        image_file.save(filepath)

        embedding = compute_face_embedding(filepath)
        if embedding is None:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({
                'success': False,
                'error': (
                    'No face detected in the image. Please upload a clear, '
                    'well-lit, front-facing face image.'
                )
            }), 400

        vector = np.asarray(embedding, dtype=np.float32)
        if vector.shape != (512,) or not np.isfinite(vector).all():
            raise ValueError('Generated face embedding must contain 512 finite values.')

        norm = float(np.linalg.norm(vector))
        if norm == 0:
            raise ValueError('Generated face embedding has zero length.')
        vector = vector / norm

        age_text = request.form.get('age', '').strip()
        age = int(age_text) if age_text.isdigit() else None



        last_seen_date = None
        date_text = request.form.get('last_seen_date', '').strip()
        if date_text:
            try:
                last_seen_date = datetime.fromisoformat(date_text)
            except ValueError:
                raise ValueError('last_seen_date must be a valid ISO date or datetime.')

        person = Person(
            name=name,
            age=age,
            gender=request.form.get('gender') or None,
            last_seen_location=request.form.get('last_seen_location') or None,
            last_seen_date=last_seen_date,
            contact_info=request.form.get('contact_info') or None,
            contact_phone=request.form.get('contact_phone') or None,
            contact_email=request.form.get('contact_email') or None,
            description=request.form.get('description') or None,
            image_filename=unique_filename,
            uploader_ip=request.remote_addr,
            source='manual_upload',
            case_status='active'
        )
        db.session.add(person)
        db.session.flush()

        face_record = FaceEmbedding(
            person_id=person.id,
            embedding=vector.astype(float).tolist(),
            embedding_dimension=512,
            is_normalized=True,
            model_name='buffalo_l',
            image_filename=unique_filename
        )
        db.session.add(face_record)
        db.session.commit()

        # Refresh the detector's in-memory cache from PostgreSQL.
        load_postgres_embeddings()

        log_audit_event(
            'PERSON_ADDED',
            f'Added missing person: {name}',
            details={'name': name, 'image_file': unique_filename, 'person_id': person.id}
        )

        return jsonify({
            'success': True,
            'name': name,
            'person_id': person.id,
            'message': f'Successfully added missing person: {name}. Person is now available for real-time detection!',
            'total_persons': Person.query.count(),
            'ready_for_detection': True
        }), 201

    except Exception as e:
        db.session.rollback()
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                logging.warning('Could not remove failed upload: %s', filepath)
        logging.exception('PostgreSQL person registration failed')
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500


@app.route('/targets', methods=['GET'])
@admin_required
def get_targets():
    """Return registered people and their embeddings from PostgreSQL."""
    try:
        persons = Person.query.order_by(Person.created_at.desc()).all()
        targets = []

        for person in persons:
            face_record = (
                FaceEmbedding.query
                .filter_by(person_id=person.id)
                .order_by(FaceEmbedding.created_at.desc())
                .first()
            )

            targets.append({
                'id': person.id,
                'name': person.name,
                'embedding': face_record.embedding if face_record else None,
                'created_at': (
                    face_record.created_at.isoformat()
                    if face_record and face_record.created_at else
                    person.created_at.isoformat() if person.created_at else None
                ),
                'image_filename': (
                    person.image_filename
                    or (face_record.image_filename if face_record else None)
                ),
                'age': person.age,
                'gender': person.gender,
                'last_seen_location': person.last_seen_location,
                'last_seen_date': person.last_seen_date.isoformat() if person.last_seen_date else None,
                'contact_info': person.contact_info,
                'contact_phone': person.contact_phone,
                'contact_email': person.contact_email,
                'description': person.description,
                'case_status': person.case_status,
                'has_embedding': face_record is not None
            })

        return jsonify({
            'success': True,
            'count': len(targets),
            'targets': targets
        }), 200

    except Exception as e:
        logging.exception('Error retrieving targets from PostgreSQL')
        return jsonify({
            'success': False,
            'error': f'Error retrieving targets: {str(e)}'
        }), 500


@app.route('/target_person/<int:person_id>', methods=['PATCH'])
@admin_required
def update_target_person(person_id):
    filepath = None

    try:
        person = db.session.get(Person, person_id)

        if not person:
            return jsonify({
                'success': False,
                'error': 'Person not found'
            }), 404

        # Update name
        name = request.form.get('name', '').strip()

        if not name:
            return jsonify({
                'success': False,
                'error': 'Name cannot be empty'
            }), 400

        duplicate = Person.query.filter(
            Person.name == name,
            Person.id != person_id
        ).first()

        if duplicate:
            return jsonify({
                'success': False,
                'error': 'Another person with this name already exists.'
            }), 409

        # Update age
        age_text = request.form.get('age', '').strip()

        if age_text:
            if not age_text.isdigit() or int(age_text) <= 0:
                return jsonify({
                    'success': False,
                    'error': 'Enter a valid age.'
                }), 400
            person.age = int(age_text)
        else:
            person.age = None

        # Update gender
        gender = request.form.get('gender', '').strip()
        if gender not in ['', 'Male', 'Female', 'Other']:
            return jsonify({
                'success': False,
                'error': 'Invalid gender.'
            }), 400

        person.gender = gender or None

        # Update last seen information
        person.last_seen_location = (
            request.form.get('last_seen_location', '').strip() or None
        )

        date_text = request.form.get('last_seen_date', '').strip()

        if date_text:
            person.last_seen_date = datetime.fromisoformat(date_text)
        else:
            person.last_seen_date = None

        # Update contact information
        person.contact_info = (
            request.form.get('contact_info', '').strip() or None
        )
        person.contact_phone = (
            request.form.get('contact_phone', '').strip() or None
        )
        person.contact_email = (
            request.form.get('contact_email', '').strip() or None
        )

        person.description = (
            request.form.get('description', '').strip() or None
        )

        # Status is editable here, not during registration
        status = request.form.get('case_status', 'active').strip().lower()

        if status not in ['active', 'under_review', 'resolved']:
            return jsonify({
                'success': False,
                'error': 'Invalid case status.'
            }), 400

        person.case_status = status

        # Replace photograph only if a new one was uploaded
        image_file = request.files.get('image')

        if image_file and image_file.filename:
            if not allowed_file(image_file.filename):
                return jsonify({
                    'success': False,
                    'error': 'Invalid image file type.'
                }), 400

            filename = secure_filename(image_file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            unique_filename = f'{timestamp}_{filename}'

            filepath = os.path.join(
                app.config['UPLOAD_FOLDER'],
                unique_filename
            )

            image_file.save(filepath)

            embedding = compute_face_embedding(filepath)

            if embedding is None:
                os.remove(filepath)
                filepath = None

                return jsonify({
                    'success': False,
                    'error': 'No face detected in the new photograph.'
                }), 400

            vector = np.asarray(embedding, dtype=np.float32)

            if vector.shape != (512,) or not np.isfinite(vector).all():
                raise ValueError('Invalid face embedding generated.')

            norm = float(np.linalg.norm(vector))

            if norm == 0:
                raise ValueError('Invalid zero-length face embedding.')

            vector = vector / norm

            face_record = (
                FaceEmbedding.query
                .filter_by(person_id=person.id)
                .order_by(FaceEmbedding.created_at.desc())
                .first()
            )

            if face_record is None:
                face_record = FaceEmbedding(
                    person_id=person.id,
                    embedding=vector.astype(float).tolist(),
                    embedding_dimension=512,
                    is_normalized=True,
                    model_name='buffalo_l'
                )
                db.session.add(face_record)
            else:
                face_record.embedding = vector.astype(float).tolist()
                face_record.embedding_dimension = 512
                face_record.is_normalized = True
                face_record.model_name = 'buffalo_l'

            face_record.image_filename = unique_filename
            person.image_filename = unique_filename

        person.name = name

        db.session.commit()

        # Refresh detection cache, including renamed people or new photos
        load_postgres_embeddings()
        log_audit_event(
            'PERSON_UPDATED',
            f'Updated missing person: {person.name}',
            details={'person_id': person.id, 'name': person.name, 'case_status': person.case_status}
        )

        return jsonify({
            'success': True,
            'message': 'Person record updated successfully.',
            'person_id': person.id
        }), 200

    except Exception as e:
        db.session.rollback()

        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass

        logging.exception('Person update failed')

        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/target_person/<name>', methods=['DELETE'])
@admin_required
def delete_target_person(name):
    """Delete a person and related embeddings/match events from PostgreSQL."""
    from urllib.parse import unquote

    person_name = unquote(name)

    try:
        person = Person.query.filter_by(name=person_name).first()
        if person is None:
            return jsonify({
                'success': False,
                'error': f'Person "{person_name}" not found in PostgreSQL'
            }), 404

        deleted_data = {
            'name': person.name,
            'image_filename': person.image_filename,
            'person_id': person.id
        }

        # MatchEvent has a non-null Person FK, so remove its rows first.
        MatchEvent.query.filter_by(person_id=person.id).delete(synchronize_session=False)

        # FaceEmbedding rows are removed by Person.embeddings delete-orphan cascade.
        db.session.delete(person)
        db.session.commit()

        load_postgres_embeddings()

        log_audit_event(
            'PERSON_DELETED',
            f'Removed missing person: {person_name}',
            details=deleted_data
        )

        return jsonify({
            'success': True,
            'message': f'Successfully removed missing person: {person_name}',
            'total_persons': Person.query.count()
        }), 200

    except Exception as e:
        db.session.rollback()
        logging.exception('PostgreSQL person deletion failed')
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500


def find_matching_person(detected_embedding, threshold=0.6):
    """Compare a supplied face embedding against embeddings loaded from PostgreSQL."""
    matches = []
    if not getattr(realtime_detector, "stored_embeddings_np", None):
        return matches

    detected = np.asarray(detected_embedding, dtype=np.float32).reshape(1, -1)
    if detected.shape[1] != 512 or not np.isfinite(detected).all():
        return matches

    for name, stored_embedding in realtime_detector.stored_embeddings_np.items():
        stored = np.asarray(stored_embedding, dtype=np.float32).reshape(1, -1)
        if stored.shape != detected.shape or not np.isfinite(stored).all():
            continue

        similarity = float(cosine_similarity(detected, stored)[0][0])
        if similarity >= threshold:
            metadata = realtime_detector.stored_embeddings.get(name, {})

            matches.append({
                "name": name,
                "similarity": similarity,
                "confidence": f"{similarity * 100:.2f}%",
                "created_at": metadata.get("created_at"),
                "image_filename": metadata.get("image_filename"),
                "person_id": metadata.get("person_id")
            })

    matches.sort(key=lambda item: item["similarity"], reverse=True)
    return matches




@app.route('/detect', methods=['POST'])
@admin_required
def detect_person():
    """
    Process an uploaded image using the separate image detection pipeline.
    Live video detection remains unchanged.
    """

    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'error': 'Missing required field: image'
        }), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected'
        }), 400

    try:
        image = read_uploaded_image(image_file)

        if image is None:
            return jsonify({
                'success': False,
                'error': 'Could not read uploaded image'
            }), 400

        result = realtime_detector.process_image(image)

        matches = result['matches']

        return jsonify({
            'success': True,
            'faces_detected': result['faces_detected'],
            'matches': matches,
            'bounding_boxes': result['bounding_boxes'],
            'message': (
                f"Found {len(matches)} potential match(es)"
                if matches
                else f"Detected {result['faces_detected']} face(s), "
                     "but no potential matches were found"
            )
        }), 200

    except Exception as e:
        logging.exception("Uploaded image detection error")

        return jsonify({
            'success': False,
            'error': f'Image processing failed: {str(e)}'
        }), 500

# ================= GROUP PHOTO FACE SELECTION =================

def read_uploaded_image(file):
    """
    Convert uploaded image into OpenCV format.
    """
    image_bytes = file.read()

    if not image_bytes:
        return None

    image_array = np.frombuffer(image_bytes, np.uint8)

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    return image


def detect_all_faces(image):
    """
    Detect all faces using the existing ArcFace model.

    Faces are sorted from left to right so that
    Face 1, Face 2, Face 3 numbering stays consistent.
    """

    faces = arcface_model.get(image)

    faces = sorted(
        faces,
        key=lambda face: float(
            (face.bbox[0] + face.bbox[2]) / 2
        )
    )

    return faces


@app.route('/detect_faces', methods=['POST'])
@admin_required
def detect_faces_for_selection():

    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'error': 'Image file is missing'
        }), 400

    try:
        image = read_uploaded_image(request.files['image'])

        if image is None:
            return jsonify({
                'success': False,
                'error': 'Could not read uploaded image'
            }), 400

        # Detect every face
        faces = detect_all_faces(image)

        face_results = []

        for index, face in enumerate(faces):

            x1, y1, x2, y2 = [
                int(round(value))
                for value in face.bbox
            ]

            face_results.append({
                'face_index': index,
                'label': f'Face {index + 1}',

                'bbox': {
                    'x1': max(0, x1),
                    'y1': max(0, y1),
                    'x2': min(image.shape[1], x2),
                    'y2': min(image.shape[0], y2)
                }
            })

        return jsonify({
            'success': True,
            'faces_detected': len(face_results),
            'faces': face_results,
            'message': f'Detected {len(face_results)} faces'
        }), 200

    except Exception as e:

        logging.exception("Face detection error")

        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/match_selected_face', methods=['POST'])
@admin_required
def match_selected_face():

    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'error': 'Image file is missing'
        }), 400

    try:
        face_index = int(request.form.get('face_index'))

        if face_index < 0:
            raise ValueError("Invalid face index")

    except (TypeError, ValueError):

        return jsonify({
            'success': False,
            'error': 'Valid face index is required'
        }), 400

    try:

        image = read_uploaded_image(request.files['image'])

        if image is None:
            return jsonify({
                'success': False,
                'error': 'Could not read uploaded image'
            }), 400

        # Detect faces again in the same left-to-right order
        faces = detect_all_faces(image)

        if face_index >= len(faces):

            return jsonify({
                'success': False,
                'error': 'Selected face not found'
            }), 400

        # Get only the selected face
        selected_face = faces[face_index]

        # Generate ArcFace embedding
        embedding = np.asarray(
            selected_face.embedding,
            dtype=np.float32
        )

        # Normalize embedding
        norm = np.linalg.norm(embedding)

        if norm == 0:
            return jsonify({
                'success': False,
                'error': 'Invalid face embedding'
            }), 400

        embedding = (embedding / norm).tolist()

        # Existing cosine similarity matching logic
        matches = realtime_detector.match_person(
            np.asarray(embedding, dtype=np.float32)
        )
        # Add registered photo metadata to each match
        for match in matches:
            metadata = realtime_detector.stored_embeddings.get(
                match.get("name"), {}
            )

            match["image_filename"] = metadata.get("image_filename")
            match["person_id"] = metadata.get("person_id")

        best_match = matches[0] if matches else None

        return jsonify({
            'success': True,
            'face_index': face_index,
            'face_label': f'Face {face_index + 1}',
            'matches': matches,
            'best_match': best_match,

            'message': (
                f"Potential match: {best_match['name']} "
                f"({best_match['confidence']})"
                if best_match
                else 'No potential match found'
            )
        }), 200

    except Exception as e:

        logging.exception("Selected face matching error")

        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Missing Person Detection API'
    }), 200


@app.route('/', methods=['GET'])
def index():
    """Public FindTrace reporting and case-tracking portal."""
    return render_template('public_portal.html')


@app.route('/app', methods=['GET'])
@admin_page_required
def legacy_application_page():
    """Existing operator-facing identification interface."""
    return render_template('index.html')


@app.route('/report', methods=['GET'])
def public_report_page():
    return render_template('report.html')


@app.route('/track', methods=['GET'])
def public_track_page():
    return render_template('track.html')


@app.route('/camera', methods=['GET'])
@admin_page_required
def camera_page():
    """Serve the camera detection page"""
    return render_template('camera.html')


@app.route('/realtime', methods=['GET'])
@admin_page_required
def realtime_page():
    """Serve the real-time detection dashboard"""
    return render_template('realtime.html')


@app.route('/admin', methods=['GET'])
def admin_page():
    """Serve the admin panel dashboard"""
    return render_template('admin_dashboard.html')


@app.route('/instructions', methods=['GET'])
def instructions_page():
    """Serve the instructions page"""
    return render_template('instructions.html')



# ================= PUBLIC CASE SUBMISSION API =================

@app.route("/api/cases/submit", methods=["POST"])
def submit_public_case():
    """Accept a public report; pending submissions do not create recognition embeddings."""
    saved_path = None
    try:
        required_text = [
            "person_name", "last_seen_location", "reporter_name", "reporter_phone"
        ]
        values = {key: request.form.get(key, "").strip() for key in required_text}
        missing = [key for key, value in values.items() if not value]
        if missing:
            return jsonify({
                "success": False,
                "error": "Required fields missing: " + ", ".join(missing)
            }), 400

        image_file = request.files.get("image")
        if not image_file or not image_file.filename:
            return jsonify({"success": False, "error": "A recent photograph is required."}), 400
        if not allowed_file(image_file.filename):
            return jsonify({"success": False, "error": "Unsupported image format."}), 400

        age_text = request.form.get("age", "").strip()
        if age_text and (not age_text.isdigit() or int(age_text) <= 0 or int(age_text) > 130):
            return jsonify({"success": False, "error": "Enter a valid age."}), 400

        date_text = request.form.get("last_seen_date", "").strip()
        last_seen_date = datetime.fromisoformat(date_text) if date_text else None

        filename = f"pending_{secrets.token_hex(12)}_{secure_filename(image_file.filename)}"
        saved_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        image_file.save(saved_path)

        # Validate that the upload is a readable image, without generating an embedding.
        if cv2.imread(saved_path) is None:
            os.remove(saved_path)
            return jsonify({"success": False, "error": "Uploaded file is not a valid image."}), 400

        case = CaseSubmission(
            case_id=create_case_id(),
            person_name=values["person_name"][:150],
            age=int(age_text) if age_text else None,
            gender=request.form.get("gender", "").strip() or None,
            last_seen_location=values["last_seen_location"][:255],
            last_seen_date=last_seen_date,
            description=request.form.get("description", "").strip() or None,
            image_filename=filename,
            reporter_name=values["reporter_name"][:150],
            reporter_phone=values["reporter_phone"][:30],
            reporter_email=request.form.get("reporter_email", "").strip() or None,
            status="pending",
        )
        db.session.add(case)
        db.session.commit()

        log_audit_event(
            "CASE_SUBMITTED",
            f"Public case submitted: {case.case_id}",
            user="public",
            details={"case_id": case.case_id, "status": "pending"}
        )
        return jsonify({
            "success": True,
            "case_id": case.case_id,
            "status": "pending",
            "message": "Your report has been submitted for admin review."
        }), 201

    except ValueError:
        db.session.rollback()
        if saved_path and os.path.exists(saved_path):
            os.remove(saved_path)
        return jsonify({"success": False, "error": "Invalid last-seen date."}), 400
    except Exception as exc:
        db.session.rollback()
        if saved_path and os.path.exists(saved_path):
            try:
                os.remove(saved_path)
            except OSError:
                pass
        logging.exception("Public case submission failed")
        return jsonify({"success": False, "error": "Could not submit the case."}), 500


@app.route("/api/cases/track/<case_id>", methods=["GET"])
def track_public_case(case_id):
    """Return safe, non-private status details for a public case ID."""
    case = CaseSubmission.query.filter_by(case_id=case_id.strip().upper()).first()
    if not case:
        return jsonify({"success": False, "error": "Case ID not found."}), 404
    return jsonify({
        "success": True,
        "case": {
            "case_id": case.case_id,
            "person_name": case.person_name,
            "status": case.status,
            "submitted_at": case.submitted_at.isoformat() if case.submitted_at else None,
            "reviewed_at": case.reviewed_at.isoformat() if case.reviewed_at else None,
            "review_note": case.review_note,
        }
    }), 200


# ================= ADMIN AUTHENTICATION =================

@app.route("/api/admin/setup", methods=["POST"])
def setup_first_admin():
    """One-time bootstrap; requires credentials configured in environment variables."""
    try:
        if Admin.query.count() > 0:
            return jsonify({"success": False, "error": "Admin setup is already completed."}), 409

        expected_key = os.getenv("ADMIN_SETUP_KEY")
        supplied_key = request.headers.get("X-Admin-Setup-Key", "")
        username = os.getenv("INITIAL_ADMIN_USERNAME", "").strip()
        password = os.getenv("INITIAL_ADMIN_PASSWORD", "")

        if not expected_key or not secrets.compare_digest(supplied_key, expected_key):
            return jsonify({"success": False, "error": "Valid setup key required."}), 403
        if not username or len(password) < 12:
            return jsonify({
                "success": False,
                "error": "Set INITIAL_ADMIN_USERNAME and an INITIAL_ADMIN_PASSWORD of at least 12 characters."
            }), 400

        admin = Admin(username=username, password_hash=generate_password_hash(password))
        db.session.add(admin)
        db.session.commit()
        log_audit_event("ADMIN_CREATED", f"Initial admin created: {username}", user=username)
        return jsonify({"success": True, "message": "Initial admin account created."}), 201
    except Exception:
        db.session.rollback()
        logging.exception("Initial admin setup failed")
        return jsonify({"success": False, "error": "Admin setup failed."}), 500


@app.route("/api/admin/login", methods=["POST"])
def admin_login():
    data = request.get_json(silent=True) or request.form
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", ""))
    admin = Admin.query.filter_by(username=username).first()
    if not admin or not check_password_hash(admin.password_hash, password):
        return jsonify({"success": False, "error": "Invalid username or password."}), 401
    session.clear()
    session["admin_id"] = admin.id
    session["admin_username"] = admin.username
    log_audit_event("ADMIN_LOGIN", f"Admin logged in: {admin.username}", user=admin.username)
    return jsonify({"success": True, "username": admin.username}), 200


@app.route("/api/admin/logout", methods=["POST"])
@admin_required
def admin_logout():
    username = session.get("admin_username", "admin")
    session.clear()
    log_audit_event("ADMIN_LOGOUT", f"Admin logged out: {username}", user=username)
    return jsonify({"success": True, "message": "Logged out."}), 200


@app.route("/api/admin/session", methods=["GET"])
@admin_required
def admin_session():
    return jsonify({"success": True, "username": session.get("admin_username")}), 200


# ================= ADMIN CASE REVIEW =================

@app.route("/api/admin/cases", methods=["GET"])
@admin_required
def admin_list_cases():
    status = request.args.get("status", "").strip().lower()
    query = CaseSubmission.query
    if status in {"pending", "approved", "rejected"}:
        query = query.filter_by(status=status)
    cases = query.order_by(CaseSubmission.submitted_at.desc()).all()
    return jsonify({
        "success": True,
        "count": len(cases),
        "cases": [case_submission_payload(case, include_private=True) for case in cases]
    }), 200


@app.route("/api/admin/cases/<case_id>/review", methods=["POST"])
@admin_required
def review_case(case_id):
    """Approve or reject a submission; approval is the only path that creates an embedding."""
    data = request.get_json(silent=True) or {}
    decision = str(data.get("decision", "")).strip().lower()
    review_note = str(data.get("review_note", "")).strip() or None
    if decision not in {"approve", "reject"}:
        return jsonify({"success": False, "error": "Decision must be approve or reject."}), 400

    case = CaseSubmission.query.filter_by(case_id=case_id.strip().upper()).with_for_update().first()
    if not case:
        return jsonify({"success": False, "error": "Case not found."}), 404
    if case.status != "pending":
        return jsonify({"success": False, "error": f"Case is already {case.status}."}), 409

    admin_name = session.get("admin_username", "admin")
    created_person = None
    try:
        if decision == "reject":
            case.status = "rejected"
            case.review_note = review_note
            case.reviewed_by = admin_name
            case.reviewed_at = datetime.utcnow()
            db.session.commit()
            log_audit_event("CASE_REJECTED", f"Case rejected: {case.case_id}", user=admin_name)
            return jsonify({"success": True, "status": "rejected", "case_id": case.case_id}), 200

        # Do not create a duplicate active Person record for an identical name.
        existing = Person.query.filter_by(name=case.person_name).first()
        if existing:
            return jsonify({
                "success": False,
                "error": "A person with this name is already registered. Review the existing record before approving."
            }), 409

        image_path = os.path.join(app.config["UPLOAD_FOLDER"], case.image_filename)
        embedding = compute_face_embedding(image_path)
        if embedding is None:
            return jsonify({
                "success": False,
                "error": "No face could be extracted from the submitted photo. Ask the reporter for a clearer image."
            }), 400

        vector = np.asarray(embedding, dtype=np.float32)
        if vector.shape != (512,) or not np.isfinite(vector).all():
            raise ValueError("Generated embedding is not a valid 512-dimensional vector.")
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            raise ValueError("Generated embedding has zero length.")
        vector = vector / norm

        created_person = Person(
            name=case.person_name,
            age=case.age,
            gender=case.gender,
            last_seen_location=case.last_seen_location,
            last_seen_date=case.last_seen_date,
            description=case.description,
            image_filename=case.image_filename,
            uploader_ip=request.remote_addr,
            source="approved_public_case",
            case_status="active",
        )
        db.session.add(created_person)
        db.session.flush()

        db.session.add(FaceEmbedding(
            person_id=created_person.id,
            embedding=vector.astype(float).tolist(),
            embedding_dimension=512,
            is_normalized=True,
            model_name="buffalo_l",
            image_filename=case.image_filename,
        ))
        case.status = "approved"
        case.review_note = review_note
        case.reviewed_by = admin_name
        case.reviewed_at = datetime.utcnow()
        case.person_id = created_person.id
        db.session.commit()

        # Only after successful DB commit is the new approved person loaded into CCTV matching.
        load_postgres_embeddings()
        log_audit_event(
            "CASE_APPROVED",
            f"Case approved and activated: {case.case_id}",
            user=admin_name,
            details={"case_id": case.case_id, "person_id": created_person.id}
        )
        return jsonify({
            "success": True,
            "status": "approved",
            "case_id": case.case_id,
            "person_id": created_person.id,
            "message": "Case approved and person added to recognition."
        }), 200

    except Exception:
        db.session.rollback()
        logging.exception("Case review failed for %s", case_id)
        return jsonify({"success": False, "error": "Could not complete case review."}), 500


@app.route("/api/admin/dashboard-summary", methods=["GET"])
@admin_required
def admin_dashboard_summary():
    return jsonify({
        "success": True,
        "pending_cases": CaseSubmission.query.filter_by(status="pending").count(),
        "approved_cases": CaseSubmission.query.filter_by(status="approved").count(),
        "rejected_cases": CaseSubmission.query.filter_by(status="rejected").count(),
        "registered_persons": Person.query.count(),
        "match_events": MatchEvent.query.count(),
        "cameras": Camera.query.count(),
    }), 200



def load_configured_cameras():
    """Load CCTV/Drone cameras from streams/cameras.json."""
    try:
        if os.path.exists(CAMERAS_CONFIG_FILE):
            with open(CAMERAS_CONFIG_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('cameras', [])
    except Exception as e:
        print(f"Error loading cameras config: {e}")
    return []


def save_configured_cameras(cameras_list):
    """Save CCTV/Drone cameras to streams/cameras.json."""
    try:
        os.makedirs(STREAMS_DIR, exist_ok=True)
        with open(CAMERAS_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump({'cameras': cameras_list}, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving cameras config: {e}")
    return False


@app.route('/api/stream/start', methods=['POST'])
@admin_required
def start_stream():
    """Start real-time video stream processing with enhanced detection"""
    global processing_thread, stream_processor
    
    data = request.json or {}
    source = data.get('source', 0)  # 0 for default webcam, or RTSP URL, IP camera, etc.
    # Ensure webcam index is int (frontend may send string "0")
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    camera_name = data.get('camera_name', str(source))  # Human-readable camera name    
    camera_location = data.get('location')
    camera_type = data.get('type')
    camera_record_id = None
    try:
        camera_record = get_or_create_camera_record(
            camera_name, source, camera_type=camera_type, location=camera_location
        )
        db.session.commit()
        camera_record_id = camera_record.id

        # Set detection backend for robustness (e.g. RetinaFace for CCTV/drone)
       
        
        # Stop any existing stream first - with proper cleanup
        if stream_processor.is_processing:
            print(f"Stopping existing stream from: {stream_processor.camera_name}")
            stream_processor.stop()
            time.sleep(1.5)  # Increased wait time to ensure full cleanup and camera release
            
            # Wait for processing thread to finish
            if processing_thread and processing_thread.is_alive():
                processing_thread.join(timeout=3)
                time.sleep(0.5)  # Extra time for OS to release camera resource
        
        # Reload the current PostgreSQL embeddings before starting the stream.
        load_postgres_embeddings()
        
        # Open camera source with name
        if stream_processor.open_camera(source, camera_name=camera_name):
            # Start processing in background thread
            def process_with_callback():

                def callback(result, annotated_frame):

                    matches = result.get('matches', [])

                    # ---------- ALERT SYSTEM (only on strong matches) ----------
                    if matches:
                        for match in matches:
                            sim = match.get('similarity', 0)
                            
                            # Only process alerts for strong matches (reduce spam)
                            if sim >= alert_system.alert_threshold:
                                # Alert system has its own cooldown - will skip repeated alerts
                                alert_record = alert_system.send_alert(
                                    match,
                                    frame=annotated_frame,
                                    frame_info={
                                        'frame': result.get('frame', 0),
                                        'faces_detected': result.get('faces_detected', 0),
                                        'total_matches': len(matches),
                                        'match_type': 'strong' if sim >= 0.55 else 'weak'
                                    },
                                    camera_source=camera_name
                                )
                                if alert_record:
                                    match['_snapshot_path'] = alert_record.get('snapshot_path')

                    # ---------- SEND MATCH DATA (deduplicated/throttled) ----------
                    if matches:
                        deduped_matches = []
                        seen_names = set()
                        now_ts = datetime.now().timestamp()

                        for match in matches:
                            name = match.get('name', 'unknown')
                            if name in seen_names:
                                continue
                            seen_names.add(name)

                            key = f"{name}_{camera_name}"
                            last_emitted = match_last_emitted.get(key, 0)
                            if now_ts - last_emitted >= match_emit_cooldown:
                                deduped_matches.append(match)
                                match_last_emitted[key] = now_ts
                                save_match_event(
                                    match,
                                    camera_source=camera_name,
                                    camera_id=camera_record_id,
                                    snapshot_path=match.get('_snapshot_path')
                                )

                        if deduped_matches:
                            log_audit_event(
                                'MATCH_DETECTED',
                                f"{len(deduped_matches)} potential match(es) detected by {camera_name}",
                                details={
                                    'camera_id': camera_record_id,
                                    'camera_source': camera_name,
                                    'person_names': [m.get('name') for m in deduped_matches]
                                }
                            )
                            socketio.emit('match_found', {
                                'matches': deduped_matches,
                                'timestamp': datetime.now().isoformat(),
                                'faces_detected': result.get('faces_detected', 0),
                                'camera_source': camera_name
                            })

                    # ---------- SEND FRAME ----------
                    try:
                        ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if ret:
                            frame_base64 = base64.b64encode(buffer).decode('utf-8')
                            socketio.emit('frame_update', {
                                'frame': frame_base64,
                                'timestamp': datetime.now().isoformat(),
                                'camera_source': camera_name,
                                'fps': getattr(stream_processor, 'fps', 0.0)
                            })
                    except Exception as e:
                        print(f"[STREAM ERROR] Frame encoding failed: {e}")

                stream_processor.process_stream(callback)

            
            set_camera_status(source, "active", is_active=True)
            processing_thread = threading.Thread(target=process_with_callback, daemon=True)
            processing_thread.start()

            log_audit_event(
                'CAMERA_STREAM_STARTED',
                f"Camera stream started: {camera_name}",
                details={'camera_id': camera_record_id, 'source': str(source)}
            )
            
            return jsonify({
                'success': True,
                'message': f'Stream started from source: {source}',
                'camera_name': camera_name,
                'stored_persons': len(realtime_detector.stored_embeddings)
            }), 200
        else:
            set_camera_status(source, "configured", is_active=True)
            return jsonify({
                'success': False,
                'error': 'Failed to open camera source'
            }), 400
    except Exception as e:
        error_msg = str(e).encode('ascii', 'replace').decode('ascii')
        print(f"Error starting stream: {error_msg}")
        return jsonify({
            'success': False,
            'error': f'{error_msg}. Make sure the camera is not in use by another application and you have proper permissions.',
            'details': str(e)
        }), 500


@app.route('/api/stream/stop', methods=['POST'])
@admin_required
def stop_stream():
    global stream_processor

    try:
        current_source = getattr(stream_processor, 'current_source', None) if stream_processor else None
        current_camera_name = getattr(stream_processor, 'camera_name', None) if stream_processor else None

        if stream_processor:
            stream_processor.stop()

        if current_source is not None:
            set_camera_status(current_source, "configured", is_active=True)
            log_audit_event(
                'CAMERA_STREAM_STOPPED',
                f"Camera stream stopped: {current_camera_name or current_source}",
                details={'source': str(current_source)}
            )

        socketio.emit('stream_stopped', {'success': True})
        return jsonify({'success': True})

    except Exception as e:
        print("STOP ERROR:", e)
        return jsonify({'success': False, 'error': str(e)}), 200

    except Exception as e:
        error_msg = str(e).encode('ascii', 'replace').decode('ascii')
        return jsonify({
            'success': False,
            'error': f'Error stopping stream: {error_msg}'
        }), 500


@app.route('/api/stream/status', methods=['GET'])
@admin_required
def stream_status():
    """Get current stream status"""
    return jsonify({
        'success': True,
        'is_processing': stream_processor.is_processing,
        'current_source': stream_processor.current_source,
        'detection_backend': getattr(realtime_detector, 'detection_backend', 'opencv'),
        'stored_persons': len(realtime_detector.stored_embeddings),
        'recent_matches': realtime_detector.get_recent_matches(5),
        'fps': getattr(stream_processor, 'fps', 0.0),
        'last_frame_time': getattr(stream_processor, 'last_frame_time', None),
        'health_status': getattr(stream_processor, 'health_status', 'unknown')
    }), 200


@app.route('/api/stream/reload', methods=['POST'])
@admin_required
def reload_embeddings():
    """Reload stored embeddings"""
    try:
        loaded_count = load_postgres_embeddings()
        return jsonify({
            'success': True,
            'message': f'Reloaded {loaded_count} embeddings'
        }), 200
    except Exception as e:
        error_msg = str(e).encode('ascii', 'replace').decode('ascii')
        return jsonify({
            'success': False,
            'error': f'Error reloading: {error_msg}'
        }), 500


@app.route('/api/alerts/history', methods=['GET'])
@admin_required
def get_alert_history():
    """Get alert history"""
    try:
        limit = request.args.get('limit', 50, type=int)
        person_name = request.args.get('person_name', None)
        
        history = alert_system.get_alert_history(limit=limit, person_name=person_name)
        
        return jsonify({
            'success': True,
            'count': len(history),
            'alerts': history
        }), 200
    except Exception as e:
        error_msg = str(e).encode('ascii', 'replace').decode('ascii')
        return jsonify({
            'success': False,
            'error': f'Error retrieving alert history: {error_msg}'
        }), 500


@app.route('/api/snapshots/<filename>', methods=['GET'])
@admin_required
def get_snapshot(filename):
    """Get snapshot image"""
    try:
        filepath = alert_system.get_snapshot(filename)
        if filepath and os.path.exists(filepath):
            return Response(
                open(filepath, 'rb').read(),
                mimetype='image/jpeg',
                headers={'Content-Disposition': f'inline; filename={filename}'}
            )
        else:
            return jsonify({
                'success': False,
                'error': 'Snapshot not found'
            }), 404
    except Exception as e:
        error_msg = str(e).encode('ascii', 'replace').decode('ascii')
        return jsonify({
            'success': False,
            'error': f'Error retrieving snapshot: {error_msg}'
        }), 500


@app.route('/api/cameras', methods=['GET'])
@admin_required
def list_cameras():
    """List available webcams and configured cameras stored in PostgreSQL."""
    try:
        available_cameras = []
        for i in range(4):
            test_cap = None
            try:
                test_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if sys.platform == 'win32' else cv2.VideoCapture(i)
                if test_cap.isOpened():
                    ret, frame = test_cap.read()
                    if ret:
                        available_cameras.append({
                            'id': f'webcam_{i}', 'name': f'Webcam {i}',
                            'source': i, 'type': 'webcam', 'status': 'available'
                        })
            except Exception:
                pass
            finally:
                if test_cap:
                    test_cap.release()

        if not any(c.get('type') == 'webcam' for c in available_cameras):
            available_cameras.insert(0, {
                'id': 'webcam_0', 'name': 'Default Webcam (try this)',
                'source': 0, 'type': 'webcam',
                'status': 'unknown - may need testing'
            })

        # One-way compatibility import: bring any older cameras.json entries
        # into PostgreSQL without creating duplicates (source is unique).
        legacy_cameras = load_configured_cameras()
        imported_legacy = False
        for legacy in legacy_cameras:
            legacy_source = legacy.get('source') or legacy.get('url')
            if legacy_source is not None and str(legacy_source) != '':
                get_or_create_camera_record(
                    legacy.get('name') or legacy.get('camera_name') or str(legacy_source),
                    legacy_source,
                    camera_type=legacy.get('type') or 'cctv'
                )
                imported_legacy = True
        if imported_legacy:
            db.session.commit()

        for cam in Camera.query.order_by(Camera.created_at.desc()).all():
            available_cameras.append({
                'id': cam.id, 'name': cam.camera_name, 'source': cam.source,
                'type': cam.camera_type, 'status': cam.status
            })

        return jsonify({
            'success': True, 'cameras': available_cameras,
            'count': len(available_cameras),
            'message': f'Found {len(available_cameras)} camera(s)'
        }), 200
    except Exception as e:
        db.session.rollback()
        logging.exception("Error listing cameras from PostgreSQL")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/cameras/save', methods=['POST'])
@admin_required
def save_camera():
    """Save a CCTV/IP/drone camera directly to PostgreSQL."""
    try:
        data = request.get_json(silent=True) or {}
        name = (data.get('name') or data.get('camera_name') or '').strip()
        source = data.get('source') or data.get('url', '')
        stream_type = (data.get('type') or 'cctv').lower()
        location = (data.get('location') or '').strip() or None

        if not name or source == '':
            return jsonify({'success': False, 'error': 'name and source (or url) are required'}), 400

        camera = get_or_create_camera_record(name, source, camera_type=stream_type, location=location)
        db.session.commit()
        log_audit_event(
            'CAMERA_ADDED', f'Added camera: {camera.camera_name}',
            details={'camera_id': camera.id, 'camera_type': camera.camera_type, 'location': camera.location}
        )

        return jsonify({
            'success': True,
            'message': f'Added {camera.camera_type}: {camera.camera_name} to PostgreSQL',
            'camera': {
                'id': camera.id, 'name': camera.camera_name, 'source': camera.source,
                'type': camera.camera_type, 'status': camera.status
            },
            'cameras': [{
                'id': c.id, 'name': c.camera_name, 'source': c.source,
                'type': c.camera_type, 'status': c.status
            } for c in Camera.query.order_by(Camera.created_at.desc()).all()]
        }), 200
    except Exception as e:
        db.session.rollback()
        logging.exception("Failed to save camera in PostgreSQL")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/cameras/test', methods=['POST'])
@admin_required
def test_camera():
    """Test if a camera source can be opened"""
    try:
        data = request.json or {}
        source = data.get('source', 0)
        
        test_cap = None
        try:
            if isinstance(source, str):
                test_cap = cv2.VideoCapture(source)
            else:
                if sys.platform == 'win32':
                    test_cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
                else:
                    test_cap = cv2.VideoCapture(int(source))
            
            if test_cap.isOpened():
                ret, frame = test_cap.read()
                if ret:
                    test_cap.release()
                    return jsonify({
                        'success': True,
                        'message': f'Camera {source} is available and working',
                        'source': source
                    }), 200
                else:
                    test_cap.release()
                    return jsonify({
                        'success': False,
                        'error': f'Camera {source} opened but cannot read frames'
                    }), 400
            else:
                if test_cap:
                    test_cap.release()
                return jsonify({
                    'success': False,
                    'error': f'Cannot open camera {source}. Make sure it is not in use by another application.'
                }), 400
        except Exception as e:
            if test_cap:
                test_cap.release()
            raise e
            
    except Exception as e:
        error_msg = str(e).encode('ascii', 'replace').decode('ascii')
        return jsonify({
            'success': False,
            'error': f'Error testing camera: {error_msg}'
        }), 500


@app.route('/api/evidence', methods=['GET'])
@admin_required
def get_evidence():
    """Get evidence collection (snapshots and alerts)"""
    try:
        limit = request.args.get('limit', 100, type=int)
        person_name = request.args.get('person_name', None)
        start_date = request.args.get('start_date', None)
        end_date = request.args.get('end_date', None)
        
        # Get alerts as evidence
        alerts = alert_system.get_alert_history(limit=limit, person_name=person_name)
        
        # Filter by date if provided
        if start_date or end_date:
            filtered_alerts = []
            for alert in alerts:
                alert_time = datetime.fromisoformat(alert['timestamp'])
                if start_date:
                    start = datetime.fromisoformat(start_date)
                    if alert_time < start:
                        continue
                if end_date:
                    end = datetime.fromisoformat(end_date)
                    if alert_time > end:
                        continue
                filtered_alerts.append(alert)
            alerts = filtered_alerts
        
        return jsonify({
            'success': True,
            'count': len(alerts),
            'evidence': alerts
        }), 200
    except Exception as e:
        error_msg = str(e).encode('ascii', 'replace').decode('ascii')
        return jsonify({
            'success': False,
            'error': f'Error retrieving evidence: {error_msg}'
        }), 500


def log_audit_event(event_type, description, user=None, details=None):
    """Persist application audit events in PostgreSQL."""
    try:
        with app.app_context():
            entry = AuditLog(
                event_type=str(event_type)[:100],
                description=str(description),
                user=str(user or "system")[:150],
                ip_address=None,
                action_status="success",
                details=details or {},
                timestamp=datetime.utcnow()
            )
            db.session.add(entry)
            db.session.commit()
        logging.info("AUDIT: %s - %s", event_type, description)
        return True
    except Exception:
        try:
            with app.app_context():
                db.session.rollback()
        except Exception:
            pass
        logging.exception("Error saving audit event to PostgreSQL")
        return False


@app.route('/api/audit', methods=['GET'])
@admin_required
def get_audit_log():
    """Retrieve audit events from PostgreSQL."""
    try:
        limit = max(1, min(request.args.get('limit', 100, type=int), 1000))
        event_type = request.args.get('event_type')
        query = AuditLog.query
        if event_type:
            query = query.filter(AuditLog.event_type == event_type)
        entries = query.order_by(AuditLog.timestamp.desc()).limit(limit).all()

        audit_entries = [{
            'id': entry.id,
            'timestamp': entry.timestamp.isoformat() if entry.timestamp else None,
            'event_type': entry.event_type,
            'description': entry.description,
            'user': entry.user,
            'ip_address': entry.ip_address,
            'action_status': entry.action_status,
            'details': entry.details or {}
        } for entry in entries]

        return jsonify({'success': True, 'count': len(audit_entries), 'audit_log': audit_entries}), 200
    except Exception as e:
        logging.exception("Error retrieving audit log from PostgreSQL")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/match-events', methods=['GET'])
@admin_required
def get_match_events():
    """Retrieve persisted face-match events from PostgreSQL."""
    try:
        limit = max(1, min(request.args.get('limit', 100, type=int), 1000))
        person_id = request.args.get('person_id', type=int)
        query = MatchEvent.query
        if person_id is not None:
            query = query.filter(MatchEvent.person_id == person_id)
        events = query.order_by(MatchEvent.detected_at.desc()).limit(limit).all()

        results = []
        for event in events:
            person = db.session.get(Person, event.person_id)
            results.append({
                'id': event.id, 'person_id': event.person_id,
                'person_name': person.name if person else None,
                'camera_id': event.camera_id,
                'camera_name': event.camera.camera_name if event.camera else None,
                'similarity': event.similarity, 'camera_source': event.camera_source,
                'snapshot_path': event.snapshot_path, 'review_status': event.review_status,
                'detected_at': event.detected_at.isoformat() if event.detected_at else None
            })
        return jsonify({'success': True, 'count': len(results), 'match_events': results}), 200
    except Exception as e:
        logging.exception("Error retrieving match events from PostgreSQL")
        return jsonify({'success': False, 'error': str(e)}), 500


@socketio.on('connect')
def handle_connect():
    """Allow real-time monitoring sockets only for authenticated admins."""
    if not session.get("admin_id"):
        return False
    emit('connected', {'message': 'Connected to real-time detection server'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    print('Client disconnected')


if __name__ == '__main__':
    # Run Flask app with SocketIO
    print("Starting Missing Person Detection API...")
    print(f"Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"Embeddings file: {os.path.abspath(EMBEDDINGS_FILE)}")
    print("Real-time detection enabled")


@app.route('/api/test/yolo', methods=['GET'])
@admin_required
def test_yolo():
    """Quick test to see if YOLO detects faces on default webcam"""
    try:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({'success': False, 'error': 'Cannot read from webcam'}), 400
        
        faces = realtime_detector.detect_faces(frame)
        return jsonify({
            'success': True,
            'faces_detected': len(faces),
            'frame_shape': frame.shape,
            'message': f'YOLO detected {len(faces)} face(s)'
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/test/embedding', methods=['GET'])
@admin_required
def test_embedding():
    """Test face embedding matching against stored persons"""
    try:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({'success': False, 'error': 'Cannot read from webcam'}), 400
        
        # Detect face
        faces = realtime_detector.detect_faces(frame)
        if len(faces) == 0:
            return jsonify({'success': False, 'error': 'No face detected by YOLO'}), 400
        
        # Get embedding for first face
        face_img = faces[0]['face']
        embedding = realtime_detector.get_embedding(face_img)
        
        if embedding is None:
            return jsonify({'success': False, 'error': 'ArcFace failed to generate embedding'}), 400
        
        # Compare with embeddings loaded from PostgreSQL.
        results = {}
        live_emb = np.asarray(embedding, dtype=np.float32).reshape(1, -1)

        for name, stored_emb in realtime_detector.stored_embeddings_np.items():
            stored_emb = np.asarray(stored_emb, dtype=np.float32).reshape(1, -1)
            similarity = float(cosine_similarity(live_emb, stored_emb)[0][0])
            results[name] = {
                'similarity': similarity,
                'percentage': f'{similarity*100:.2f}%',
                'matches': similarity >= 0.30
            }
        
        return jsonify({
            'success': True,
            'message': 'Embedding comparison complete',
            'face_detected': True,
            'embedding_generated': True,
            'stored_persons': results,
            'threshold': 0.30
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == "__main__":
    socketio.run(
        app,
        host="0.0.0.0",
        port=5001,
        debug=False,
        allow_unsafe_werkzeug=True
    )

