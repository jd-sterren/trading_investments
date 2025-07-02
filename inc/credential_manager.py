import os, sys
import base64
import shutil
import getpass
from dotenv import dotenv_values
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

# ==== UTILITY FUNCTIONS ====

def generate_salt(length=16):
    return os.urandom(length)

def save_salt(salt, filename):
    with open(filename, "wb") as f:
        f.write(salt)

def load_salt(filename):
    with open(filename, "rb") as f:
        return f.read()

def derive_key(passphrase: str, salt: bytes) -> bytes:
    passphrase_bytes = passphrase.encode()
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100_000,
    )
    return kdf.derive(passphrase_bytes)

def backup_file(filepath):
    """Create a .bak backup before modifying."""
    if os.path.exists(filepath):
        shutil.copy(filepath, filepath + ".bak")
        print(f"Backup created: {filepath}.bak")

def get_paths(env="dev"):
    """Return correct paths for credentials and salt files."""
    base_dir = os.path.dirname(__file__)  # this gets /inc automatically
    cred_dir = os.path.join(base_dir, "credentials", env)
    credentials_file = os.path.join(cred_dir, "credentials.env")
    salt_file = os.path.join(cred_dir, "credentials.salt")
    return credentials_file, salt_file

def ensure_env_folder(env="dev"):
    base_dir = os.path.join("credentials", env)
    os.makedirs(base_dir, exist_ok=True)

# ==== AES-GCM ENCRYPT/DECRYPT ====

def encrypt_value(key, plaintext):
    nonce = os.urandom(12)  # 96-bit nonce
    encryptor = Cipher(
        algorithms.AES(key),
        modes.GCM(nonce)
    ).encryptor()

    ciphertext = encryptor.update(plaintext.encode()) + encryptor.finalize()
    # Return all together: nonce + tag + ciphertext
    return base64.urlsafe_b64encode(nonce + encryptor.tag + ciphertext).decode()

def decrypt_value(key, token):
    decoded = base64.urlsafe_b64decode(token.encode())
    nonce = decoded[:12]
    tag = decoded[12:28]
    ciphertext = decoded[28:]

    decryptor = Cipher(
        algorithms.AES(key),
        modes.GCM(nonce, tag)
    ).decryptor()

    plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    return plaintext.decode()

def inject_decrypted_env(environment="dev", required_vars=None, crash_on_fail=True, passphrase=None):
    """
    Decrypts environment variables and injects them into os.environ.
    
    Args:
        environment (str): The environment to load ('dev', 'staging', 'prod').
        required_vars (list, optional): List of required variable names.
        crash_on_fail (bool, optional): Whether to exit if required variables are missing.
        passphrase (str, optional): Passphrase for decryption. If None, default behavior is used.
    """
    try:
        env_vars = decrypt_variables(environment=environment, passphrase=passphrase)
    except Exception as e:
        print(f"Failed to decrypt environment '{environment}': {e}")
        if crash_on_fail:
            exit(1)
        else:
            return False

    for key, value in env_vars.items():
        os.environ[key] = value

    if required_vars:
        missing = []
        for var in required_vars:
            if var not in os.environ or not os.environ[var]:
                missing.append(var)

        if missing:
            print(f"Missing required environment variables: {', '.join(missing)}")
            if crash_on_fail:
                exit(1)
            else:
                return False

    return True  # Successfully injected



# ==== MAIN FUNCTIONS ====

def add_encrypted_variables(environment="dev"):
    ensure_env_folder(environment)
    credentials_file, salt_file = get_paths(environment)

    file_exists = os.path.exists(credentials_file)
    salt_exists = os.path.exists(salt_file)

    if not salt_exists:
        print(f"Generating new salt for environment '{environment}'...")
        salt = generate_salt()
        save_salt(salt, salt_file)
    else:
        salt = load_salt(salt_file)

    if file_exists:
        env_vars = dotenv_values(credentials_file)
    else:
        env_vars = {}

    passphrase = getpass.getpass("Enter a passphrase to encrypt your environment file: ")
    key = derive_key(passphrase, salt)

    while True:
        var_name = input("Enter the variable name (e.g., OPENAI_API_KEY): ").strip()
        var_value = input(f"Enter the value for {var_name}: ").strip()

        if var_name in env_vars:
            print(f"Variable '{var_name}' already exists. It will be overwritten.")

        encrypted_value = encrypt_value(key, var_value)
        env_vars[var_name] = encrypted_value

        another = input("Add another variable? (y/n): ").lower()
        if another != 'y':
            break

    backup_file(credentials_file)

    with open(credentials_file, "w") as file:
        for var, value in env_vars.items():
            file.write(f"{var}={value}\n")

    print(f"Encrypted variables saved to {credentials_file}")

import sys  # Add this at the top if not already

def decrypt_variables(environment="dev", passphrase=None):
    """Decrypts environment variables from an encrypted file and returns them as a dictionary."""
    credentials_file, salt_file = get_paths(environment)

    if not os.path.exists(credentials_file) or not os.path.exists(salt_file):
        raise FileNotFoundError(f"Environment '{environment}' is missing credentials or salt file.")

    salt = load_salt(salt_file)

    # Allow up to 3 attempts for correct passphrase
    attempts = 0
    max_attempts = 3

    while attempts < max_attempts:
        if passphrase is None:
            passphrase = getpass.getpass("Enter your passphrase to decrypt environment variables: ")

        key = derive_key(passphrase, salt)
        encrypted_env_vars = dotenv_values(credentials_file)

        # Try decrypting at least ONE variable to verify passphrase
        test_var = next(iter(encrypted_env_vars.values()), None)
        if test_var:
            try:
                _ = decrypt_value(key, test_var)
                break  # Passphrase is correct, exit loop
            except Exception:
                attempts += 1
                print(f"Incorrect passphrase ({attempts}/{max_attempts} attempts)")
                passphrase = None  # Reset passphrase for next attempt
                if attempts >= max_attempts:
                    print("Too many failed attempts. Exiting for security.")
                    sys.exit(1)

    # Decrypt all variables after passphrase verified
    decrypted_env_vars = {}
    for var, value in encrypted_env_vars.items():
        try:
            decrypted_value = decrypt_value(key, value)
            decrypted_env_vars[var] = decrypted_value
        except Exception:
            decrypted_env_vars[var] = value  # Fallback if specific var fails

    return decrypted_env_vars


def change_passphrase(environment="dev"):
    credentials_file, salt_file = get_paths(environment)

    if not os.path.exists(credentials_file) or not os.path.exists(salt_file):
        raise FileNotFoundError(f"Environment '{environment}' is missing credentials or salt file.")

    print(f"Decrypting existing variables for environment '{environment}'...")
    old_passphrase = getpass.getpass("Enter the OLD passphrase: ")
    old_salt = load_salt(salt_file)
    old_key = derive_key(old_passphrase, old_salt)

    encrypted_env_vars = dotenv_values(credentials_file)
    decrypted_env_vars = {}

    for var, value in encrypted_env_vars.items():
        try:
            decrypted_value = decrypt_value(old_key, value)
            decrypted_env_vars[var] = decrypted_value
        except Exception as e:
            print(f"Error decrypting variable '{var}': {e}")
            return  # Abort to avoid partial re-encryption

    print(f"Now setting a new passphrase...")
    new_passphrase = getpass.getpass("Enter the NEW passphrase: ")
    confirm_passphrase = getpass.getpass("Confirm the NEW passphrase: ")

    if new_passphrase != confirm_passphrase:
        print("New passphrases do not match. Aborting.")
        return

    new_salt = generate_salt()
    save_salt(new_salt, salt_file)
    new_key = derive_key(new_passphrase, new_salt)

    backup_file(credentials_file)
    backup_file(salt_file)

    with open(credentials_file, "w") as file:
        for var, value in decrypted_env_vars.items():
            encrypted_value = encrypt_value(new_key, value)
            file.write(f"{var}={encrypted_value}\n")

    print(f"Passphrase changed successfully for environment '{environment}'!")

# ==== CLI MENU ====

def main():
    print("\nWelcome to Credential Manager (AES-GCM Edition)")

    while True:
        print("\nChoose an option:")
        print("1. Add Encrypted Variables")
        print("2. Decrypt and View Variables")
        print("3. Change Passphrase")
        print("4. Exit")

        choice = input("\nEnter your choice (1/2/3/4): ").strip()

        if choice == '1':
            env = input("Enter the environment (dev/staging/prod): ").strip().lower()
            add_encrypted_variables(environment=env)
        elif choice == '2':
            env = input("Enter the environment (dev/staging/prod): ").strip().lower()
            try:
                decrypted = decrypt_variables(environment=env)
                print("\nDecrypted Variables:")
                for k, v in decrypted.items():
                    print(f"{k} = {v}")
            except FileNotFoundError as e:
                print(f"Error: {e}")
        elif choice == '3':
            env = input("Enter the environment (dev/staging/prod): ").strip().lower()
            try:
                change_passphrase(environment=env)
            except FileNotFoundError as e:
                print(f"Error: {e}")
        elif choice == '4':
            print("Exiting Credential Manager. Stay secure!")
            break
        else:
            print("Invalid choice. Please select 1, 2, 3, or 4.")

def get_passphrase(passphrase_path="inc/credentials/prod/.passphrase"):
    """
    Load passphrase from a hidden file for automation.
    """
    try:
        with open(passphrase_path, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Passphrase file not found: {passphrase_path}")
        exit(1)

if __name__ == "__main__":
    main()
