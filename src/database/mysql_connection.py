import os

import mysql.connector
from dotenv import load_dotenv
from mysql.connector import Error


load_dotenv()


def create_connection():
    try:
        connection = mysql.connector.connect(
            host=os.getenv("MYSQL_HOST"),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DATABASE")
        )

        if connection.is_connected():
            print("MySQL connection successful.")

        return connection

    except Error as e:
        print(f"MySQL connection error: {e}")
        return None


if __name__ == "__main__":
    connection = create_connection()

    if connection:
        connection.close()
        print("MySQL connection closed.")