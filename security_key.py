import time

class SecurityKey:
    def __init__(self):
        self.key = None
        self.valid_until = None  

    def generate(self):
        self.key = "secure_key" 
        self.valid_until = time.time() + 60  

    def validate(self, key):
        return key == self.key and time.time() < self.valid_until
