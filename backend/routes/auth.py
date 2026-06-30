"""
MANA — Auth Routes backed by the SQLite user store.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request
from flask_jwt_extended import (
    create_access_token,
    get_jwt_identity,
    jwt_required,
)

from data import now_utc
from models import ActivityLog, User, db

auth_bp = Blueprint("auth", __name__)


def get_json():
    return request.get_json() or {}


def normalize_identity(raw_identity: str) -> str:
    return (raw_identity or "").strip()


def find_user(identity: str):
    if not identity:
        return None
    return (
        User.query.filter(
            (User.username == identity) | (User.email == identity.lower())
        ).first()
    )


def log_activity(
    action: str,
    detail: str,
    log_type: str = "system",
    actor: User | None = None,
    target: User | None = None,
):
    actor = actor or current_user()
    actor_name = actor.name or actor.username if actor else "System"
    actor_username = actor.username if actor else None
    db.session.add(
        ActivityLog(
            actor_username=actor_username,
            actor_name=actor_name,
            action=action,
            detail=detail,
            type=log_type,
            target_username=target.username if target else None,
            target_name=(target.name or target.username) if target else None,
        )
    )


def current_user():
    username = get_jwt_identity()
    if not username:
        return None
    return db.session.get(User, username)


@auth_bp.route("/login", methods=["POST"])
def login():
    data = get_json()
    identity = normalize_identity(data.get("username") or data.get("email"))
    password = data.get("password", "")

    user = find_user(identity)
    if not user or not user.check_password(password):
        return jsonify({"message": "Invalid credentials"}), 401
    if user.status != "Active":
        return jsonify({"message": f"Account is {user.status.lower()}."}), 403

    user.login_count = (user.login_count or 0) + 1
    user.last_login_at = now_utc()

    token = create_access_token(identity=user.username, additional_claims={"role": user.role})
    log_activity("Logged in", f"Signed in to {user.role} workspace", "auth", actor=user)
    db.session.commit()

    return jsonify(
        {
            "token": token,
            "user": {
                "username": user.username,
                "name": user.name or user.username,
                "role": user.role,
                "email": user.email,
                "status": user.status,
            },
        }
    )


@auth_bp.route("/logout", methods=["POST"])
@jwt_required()
def logout():
    user = current_user()
    if user:
        log_activity("Logged out", "Signed out of MANA", "auth", actor=user)
        db.session.commit()
    return jsonify({"success": True})


@auth_bp.route("/me", methods=["GET"])
@jwt_required()
def get_profile():
    user = current_user()
    if not user:
        return jsonify({"message": "User not found"}), 404
    return jsonify(
        {
            "username": user.username,
            "name": user.name or user.username,
            "role": user.role,
            "email": user.email,
            "status": user.status,
        }
    )


@auth_bp.route("/me", methods=["PATCH"])
@jwt_required()
def update_profile():
    user = current_user()
    if not user:
        return jsonify({"message": "User not found"}), 404

    data = get_json()
    name = (data.get("name") or data.get("username") or "").strip()
    changes = []
    if name and name != (user.name or user.username):
        changes.append(f"name '{user.name or user.username}' → '{name}'")
        user.name = name

    detail = "Updated own profile — " + ("; ".join(changes) if changes else "no field changes")
    log_activity("Profile updated", detail, "edit", actor=user)
    db.session.commit()
    return jsonify(
        {
            "username": user.username,
            "name": user.name or user.username,
            "role": user.role,
            "email": user.email,
            "status": user.status,
        }
    )


@auth_bp.route("/change-password", methods=["POST"])
@jwt_required()
def change_password():
    user = current_user()
    if not user:
        return jsonify({"message": "User not found"}), 404

    data = get_json()
    current_password = data.get("current_password", "")
    new_password = data.get("new_password", "")
    confirm_password = data.get("confirm_password", "")

    if not user.check_password(current_password):
        return jsonify({"message": "Current password is incorrect"}), 400
    if len(new_password) < 8:
        return jsonify({"message": "New password must be at least 8 characters."}), 400
    if new_password != confirm_password:
        return jsonify({"message": "New password and confirmation do not match."}), 400

    user.set_password(new_password)
    log_activity("Password changed", "Updated own account password", "auth", actor=user)
    db.session.commit()
    return jsonify({"success": True})


@auth_bp.route("/request-email-change", methods=["POST"])
@jwt_required()
def request_email_change():
    return jsonify({"success": True})


@auth_bp.route("/verify-email-change", methods=["POST"])
@jwt_required()
def verify_email_change():
    user = current_user()
    if not user:
        return jsonify({"message": "User not found"}), 404

    data = get_json()
    email = (data.get("new_email") or "").strip().lower()
    code = (data.get("code") or "").strip()

    if not email:
        return jsonify({"message": "New email is required"}), 400
    if code != "246810":
        return jsonify({"message": "Invalid verification code"}), 400
    existing = User.query.filter(User.email == email, User.username != user.username).first()
    if existing:
        return jsonify({"message": "Email is already in use"}), 409

    user.email = email
    log_activity("Email updated", f"Changed email to {email}", "edit", actor=user)
    db.session.commit()
    return jsonify({"success": True, "email": email})
