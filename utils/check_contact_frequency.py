import sqlite3
import re

def extract_email(text):
    if not text:
        return ""
    match = re.search(r'<([^>]+)>', text)
    if match:
        return match.group(1)
    match = re.search(r'\b[\w.-]+@[\w.-]+\.\w+\b', text)
    if match:
        return match.group(0)
    return ""

def check_contact_frequency(db_path, contact):
    conn = sqlite3.connect(db_path)
    conn.create_function("extract_email", 1, extract_email)
    cursor = conn.cursor()
    query = """
    SELECT COUNT(*) FROM emails 
    WHERE extract_email(보낸사람) = ? OR extract_email(받은사람) = ?;
    """
    cursor.execute(query, (contact, contact))
    result = cursor.fetchone()
    conn.close()
    return result[0]

if __name__ == "__main__":
    db_path = "d:\\PythonProject\\llm\\email_analyzer2\\data\\temp_Portable_search_DB_1062개.db"
    contact = "rex@hjc-helmet.com"
    freq = check_contact_frequency(db_path, contact)
    print(f"연락처 {contact}의 빈도수: {freq}")
