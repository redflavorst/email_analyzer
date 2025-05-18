import sqlite3
import re

def extract_email(text):
    """
    텍스트에서 이메일 주소를 추출합니다.
    
    Args:
        text (str): 이메일 주소가 포함된 텍스트
    
    Returns:
        str: 추출된 이메일 주소 또는 빈 문자열
    """
    if not text:
        return ""
    match = re.search(r'<([^>]+)>', text)
    if match:
        return match.group(1)
    match = re.search(r'\b[\w.-]+@[\w.-]+\.\w+\b', text)
    if match:
        print(match.group(0))
        return match.group(0)
    return ""

def load_contact_frequencies(db_path):
    """
    SQLite DB에서 연락처별 빈도수를 로드합니다.
    
    Args:
        db_path (str): SQLite DB 파일 경로
    
    Returns:
        dict: 연락처별 빈도수 딕셔너리
    """
    frequencies = {}
    try:
        conn = sqlite3.connect(db_path)
        conn.create_function("extract_email", 1, extract_email)
        cursor = conn.cursor()
        
        query = """
        SELECT contact, COUNT(*) as frequency
        FROM (
            SELECT extract_email(보낸사람) as contact FROM emails
            UNION ALL
            SELECT extract_email(받은사람) as contact FROM emails
        ) combined
        WHERE contact IS NOT NULL AND contact != ''
        GROUP BY contact
        ORDER BY frequency DESC;
        """
        cursor.execute(query)
        results = cursor.fetchall()
        
        for row in results:
            contact = row[0]
            freq = row[1]
            if contact != '자기자신@hjc-helmet.com':  # 자기 자신 제외 (실제 이메일로 변경 필요)
                frequencies[contact] = freq
        
        conn.close()
    except sqlite3.Error as e:
        print(f"DB 연결 오류: {e}")
        return {}
    
    return frequencies

def identify_mail_owner(frequencies):
    """
    빈도수를 기준으로 메일 주인을 식별합니다.
    
    Args:
        frequencies (dict): 연락처별 빈도수 딕셔너리
    
    Returns:
        str: 메일 주인의 이메일 주소
    """
    if not frequencies:
        return None
    return max(frequencies.items(), key=lambda x: x[1])[0]

def load_contact_frequencies_with_owner(db_path):
    """
    SQLite DB에서 연락처별 빈도수를 로드하고 메일 주인을 식별합니다.
    
    Args:
        db_path (str): SQLite DB 파일 경로
    
    Returns:
        tuple: (dict, str) 연락처별 빈도수 딕셔너리와 메일 주인의 이메일 주소
    """
    frequencies = load_contact_frequencies(db_path)
    owner = identify_mail_owner(frequencies)
    return frequencies, owner

def load_opponent_frequencies(db_path, owner_email):
    """
    SQLite DB에서 메일 주인과 상대방 간의 빈도수를 로드합니다.
    
    Args:
        db_path (str): SQLite DB 파일 경로
        owner_email (str): 메일 주인의 이메일 주소
    
    Returns:
        dict: 상대방별 빈도수 딕셔너리
    """
    opponent_frequencies = {}
    try:
        conn = sqlite3.connect(db_path)
        conn.create_function("extract_email", 1, extract_email)
        cursor = conn.cursor()
        
        query = """
        SELECT 
            CASE 
                WHEN extract_email(보낸사람) = ? THEN extract_email(받은사람)
                ELSE extract_email(보낸사람)
            END as opponent, 
            COUNT(*) as frequency
        FROM emails
        WHERE extract_email(보낸사람) = ? OR extract_email(받은사람) = ?
        GROUP BY opponent
        ORDER BY frequency DESC;
        """
        cursor.execute(query, (owner_email, owner_email, owner_email))
        results = cursor.fetchall()
        
        for row in results:
            opponent = row[0]
            freq = row[1]
            if opponent and opponent != owner_email:
                opponent_frequencies[opponent] = freq
        
        conn.close()
    except sqlite3.Error as e:
        print(f"DB 연결 오류: {e}")
        return {}
    
    return opponent_frequencies

if __name__ == "__main__":
    db_path = "d:\\PythonProject\\llm\\email_analyzer2\\data\\temp_Portable_search_DB_1062개.db"
    freqs, owner = load_contact_frequencies_with_owner(db_path)
    print(f"메일 주인: {owner}")
    for contact, freq in freqs.items():
        print(f"연락처: {contact}, 빈도수: {freq}")
    opponent_freqs = load_opponent_frequencies(db_path, owner)
    print("\n상대방별 빈도수:")
    for opponent, freq in opponent_freqs.items():
        print(f"상대방: {opponent}, 빈도수: {freq}")
