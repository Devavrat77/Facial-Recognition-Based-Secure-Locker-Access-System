import sqlite3

def create_database():
    conn = sqlite3.connect('customer_data.db')
    cursor = conn.cursor()

    # Create table for customer information
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS customers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_id TEXT UNIQUE NOT NULL,
        locker_number TEXT NOT NULL,
        face_embedding BLOB NOT NULL
    )
    ''')

    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_database()
