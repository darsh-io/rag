"""Per-user Fernet encryption for chat message content."""
import os, base64
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

_SECRET_PATH = Path(__file__).parent.parent / "config" / ".server_secret"


def _master_secret() -> bytes:
    _SECRET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _SECRET_PATH.exists():
        return _SECRET_PATH.read_bytes()
    secret = os.urandom(32)
    _SECRET_PATH.write_bytes(secret)
    return secret


def _fernet(user_id: str) -> Fernet:
    key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=user_id.encode(),
        info=b"rewise-chat",
        backend=default_backend(),
    ).derive(_master_secret())
    return Fernet(base64.urlsafe_b64encode(key))


def encrypt(user_id: str, plaintext: str) -> str:
    return _fernet(user_id).encrypt(plaintext.encode()).decode()


def decrypt(user_id: str, ciphertext: str) -> str:
    return _fernet(user_id).decrypt(ciphertext.encode()).decode()
