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
        return match.group(0)
    return ""

def load_emails_for_contact(db_path, owner_email, opponent_email=None):
    """
    SQLite DB에서 특정 연락처와 관련된 이메일을 로드합니다.
    
    Args:
        db_path (str): SQLite DB 파일 경로
        owner_email (str): 메일 주인의 이메일 주소
        opponent_email (str, optional): 상대방의 이메일 주소. None이면 모든 상대방과의 이메일을 로드합니다.
    
    Returns:
        list: 이메일 데이터 리스트
    """
    emails = []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        if opponent_email:
            query = """
            SELECT 문서No, 보낸사람, 받은사람, 받은시간, 제목, 본문
            FROM emails
            WHERE (보낸사람 LIKE ? AND 받은사람 LIKE ?)
               OR (보낸사람 LIKE ? AND 받은사람 LIKE ?)
            ORDER BY 받은시간 ASC;
            """
            cursor.execute(query, (f"%{owner_email}%", f"%{opponent_email}%", f"%{opponent_email}%", f"%{owner_email}%"))
            print(f"쿼리 실행 - owner_email: {owner_email}, opponent_email: {opponent_email}")
            print(f"실행된 쿼리: {query}")
        else:
            query = """
            SELECT 문서No, 보낸사람, 받은사람, 받은시간, 제목, 본문
            FROM emails
            WHERE 보낸사람 LIKE ? OR 받은사람 LIKE ?
            ORDER BY 받은시간 ASC;
            """
            cursor.execute(query, (f"%{owner_email}%", f"%{owner_email}%"))
            print(f"쿼리 실행 - owner_email: {owner_email}")
            print(f"실행된 쿼리: {query}")
        
        results = cursor.fetchall()
        print(f"쿼리 결과 - 로드된 이메일 수: {len(results)}")
        if len(results) > 0:
            print(f"첫 번째 이메일 - 보낸사람: {results[0][1]}, 받은사람: {results[0][2]}, 제목: {results[0][4]}")
        for row in results:
            email = {
                '문서No': row[0],
                'sender': row[1],
                'receiver': row[2],
                'date': row[3],
                'subject': row[4],
                'content': row[5]
            }
            emails.append(email)
            print(f"로드된 이메일 상세 정보 - ID: {email['문서No']}, 제목: {email['subject']}, 보낸시간: {email['date']}")
        
        conn.close()
    except sqlite3.Error as e:
        print(f"DB 연결 오류: {e}")
        return []
    
    return emails

if __name__ == "__main__":
    db_path = "d:\\PythonProject\\llm\\email_analyzer2\\data\\temp_Portable_search_DB_1062개.db"
    owner_email = "example@domain.com"  # 테스트용 연락처
    opponent_email = "opponent@example.com"  # 상대방 이메일
    emails = load_emails_for_contact(db_path, owner_email, opponent_email)
    for email in emails:
        print(f"ID: {email['문서No']}, 제목: {email['subject']}, 보낸시간: {email['date']}")
