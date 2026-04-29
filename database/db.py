import pymysql


DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "sql",
    "database": "deepfake_db",
    "charset": "utf8",
    "cursorclass": pymysql.cursors.DictCursor,
}

def get_connection():
    try:
        return pymysql.connect(**DB_CONFIG)
    except RuntimeError as exc:
        message = str(exc)
        if "'cryptography' package is required" in message:
            raise RuntimeError(
                "Missing Python dependency: install 'cryptography' in the project virtualenv "
                "to connect with the current MySQL authentication method."
            ) from exc
        raise


def save_result(path, result, confidence):
    query = """
        INSERT INTO analysis (image_path, result, confidence, attack_type)
        VALUES (%s, %s, %s, %s)
    """
    attack_type = result if result in ("Deepfake", "Morphing Attack") else "None"

    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(query, (path, result, confidence, attack_type))
        connection.commit()
    finally:
        connection.close()


def get_history():
    query = "SELECT * FROM analysis ORDER BY created_at DESC"

    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall()
    finally:
        connection.close()
