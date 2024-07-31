import sqlite3
import numpy as np
import pickle  # To convert list to binary format

class Customer:
    def __init__(self, customer_id, locker_number):
        self.customer_id = customer_id
        self.locker_number = locker_number

    def _connect_db(self):
        return sqlite3.connect('customer_data.db')

    def get_stored_embedding(self):
        conn = self._connect_db()
        cursor = conn.cursor()

        # Fetch the stored face embedding vector
        cursor.execute('SELECT face_embedding FROM customers WHERE customer_id = ?', (self.customer_id,))
        result = cursor.fetchone()

        conn.close()
        if result:
            return pickle.loads(result[0])
        else:
            raise ValueError("Customer ID not found")

    def add_customer(self, face_embedding):
        conn = self._connect_db()
        cursor = conn.cursor()

        # Convert numpy array to binary
        face_embedding_blob = pickle.dumps(face_embedding)

        cursor.execute('INSERT OR REPLACE INTO customers (customer_id, locker_number, face_embedding) VALUES (?, ?, ?)',
                       (self.customer_id, self.locker_number, face_embedding_blob))

        conn.commit()
        conn.close()
