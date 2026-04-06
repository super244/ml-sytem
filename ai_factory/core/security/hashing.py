"""Secure hashing implementation for AI-Factory."""

import hashlib
import secrets


class SecureHasher:
    """Secure hashing implementation using PBKDF2."""

    @staticmethod
    def hash_data(data: str, salt: str | None = None) -> tuple[str, str]:
        """
        Hash data with proper salt using PBKDF2.

        Args:
            data: The data to hash.
            salt: Optional salt. If not provided, a secure random salt is generated.

        Returns:
            A tuple of (hash_hex, salt).
        """
        if salt is None:
            salt = secrets.token_hex(16)

        hash_bytes = hashlib.pbkdf2_hmac(
            "sha256",
            data.encode("utf-8"),
            salt.encode("utf-8"),
            100000,  # iterations
        )
        return hash_bytes.hex(), salt

    @staticmethod
    def verify_data(data: str, hash_hex: str, salt: str) -> bool:
        """
        Verify data against a hash.

        Args:
            data: The data to verify.
            hash_hex: The expected hash.
            salt: The salt used for hashing.

        Returns:
            True if the data matches the hash, False otherwise.
        """
        computed_hash, _ = SecureHasher.hash_data(data, salt)
        return secrets.compare_digest(computed_hash, hash_hex)

    @staticmethod
    def derive_key(password: str, salt: bytes | None = None, key_length: int = 32) -> tuple[bytes, bytes]:
        """
        Derive a cryptographic key from a password.

        Args:
            password: The password to derive key from.
            salt: Optional salt bytes. If not provided, generates random salt.
            key_length: Length of the derived key in bytes.

        Returns:
            A tuple of (key, salt).
        """
        if salt is None:
            salt = secrets.token_bytes(16)

        key = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            100000,
            dklen=key_length,
        )
        return key, salt
