from security_key import SecurityKey

class AccessLocker:
    def unlock(self, locker_number, key):
        if SecurityKey().validate(key):
            print(f"Locker {locker_number} unlocked.")
        else:
            print("Invalid or expired key.")
