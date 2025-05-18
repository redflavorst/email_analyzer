from utils.load_emails_for_contact import load_emails_for_contact
import logging
from bs4 import BeautifulSoup, NavigableString, Tag
import re
import sqlite3
import os
import ollama
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler('clean_and_group_emails_node.log', mode='w'), logging.StreamHandler()])

load_dotenv() # .env 파일 로드

class CleanAndGroupEmailsNode:
    """
    이메일 데이터를 로드하고 연락처별로 그룹화하는 노드
    """
    def __init__(self, db_path):
        self.db_path = db_path
        load_dotenv() # .env 파일 로드

        # Ollama 클라이언트 초기화 (다시 추가)
        try:
            self.ollama_client = ollama.Client()
            self.ollama_client.list() # 연결 확인
            logging.info("Ollama client initialized successfully.")
            print("Ollama client initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing Ollama client: {e}")
            print(f"Error initializing Ollama client: {e}. LLM pattern removal will be skipped.")
            self.ollama_client = None # 오류 시 None으로 설정

    def process(self, shared_store):
        """
        연락처별로 이메일을 로드하고 내용을 정리합니다.
        
        Args:
            shared_store (dict): 데이터를 공유하기 위한 저장소
        
        Returns:
            dict: 업데이트된 shared_store
        """
        logging.info("CleanAndGroupEmailsNode 실행 시작")
        print("CleanAndGroupEmailsNode 실행 시작")
        contact_frequencies = shared_store.get('contact_frequencies', {})
        owner_email = shared_store.get('owner_email', None)
        if not owner_email:
            logging.error("메일 주인이 식별되지 않았습니다.")
            print("메일 주인이 식별되지 않았습니다.")
            return shared_store
        
        opponent_frequencies = shared_store.get('opponent_frequencies', {})
        logging.info(f"그룹화 시작 - 상대방 수: {len(opponent_frequencies)}")
        print(f"그룹화 시작 - 상대방 수: {len(opponent_frequencies)}")
        # 연락 빈도수를 기준으로 상위 10개 상대방만 선택
        top_opponents = sorted(opponent_frequencies.items(), key=lambda x: x[1], reverse=True)[:1]
        logging.info(f"상위 1개 상대방 선택: {len(top_opponents)}")
        print(f"상위 1개 상대방 선택: {len(top_opponents)}")
        logging.info(f"상위 1개 상대방 목록: {[opponent for opponent, freq in top_opponents]}")
        print(f"상위 1개 상대방 목록: {[opponent for opponent, freq in top_opponents]}")
        grouped_emails = {}
        for opponent, freq in top_opponents:
            logging.info(f"상대방 {opponent} 처리 시작")
            print(f"상대방 {opponent} 처리 시작")
            # Use 'id' field based on memory
            emails = load_emails_for_contact(self.db_path, owner_email, opponent)

            logging.info(f"상대방 {opponent} - 로드된 이메일 수: {len(emails)}")
            print(f"상대방 {opponent} - 로드된 이메일 수: {len(emails)}")

            structurally_cleaned_emails = [] # 구조 클리닝 결과 저장 리스트
            if len(emails) > 0:
                logging.info(f"상대방 {opponent} - 첫 번째 이메일 ID: {emails[0].get('문서No', 'N/A')}") 
                print(f"상대방 {opponent} - 첫 번째 이메일 ID: {emails[0].get('문서No', 'N/A')}")
                # 디버깅: 이메일 딕셔너리 구조 확인
                #logging.info(f"이메일 딕셔너리 키: {list(emails[0].keys())}")
                #print(f"이메일 딕셔너리 키: {list(emails[0].keys())}")

                # 1. Perform initial structural cleaning for all emails
                for email in emails:
                    email_id = email.get('문서No') 
                    if not email_id:
                        logging.warning(f"Skipping email due to missing '문서No': {email.get('subject', 'No Subject')}")
                        continue
                    email['문서No'] = str(email_id) # Ensure ID is string, using '문서No'

                    raw_content = email.get('content', '') # Use 'content'
                    logging.info(f"Email ID {email.get('문서No')} - Calling clean_email_content with raw_content length: {len(raw_content)}")
                    cleaned_content = self.clean_email_content(raw_content)
                    logging.info(f"Email ID {email.get('문서No')} - Returned cleaned_content length: {len(cleaned_content)}")
                    email['cleaned_content'] = cleaned_content # Store structurally cleaned content
                    structurally_cleaned_emails.append(email)

            # 3. Write final cleaned content to TXT file
            if structurally_cleaned_emails:
                txt_file_path = f"d:\\PythonProject\\llm\\email_analyzer2\\data\\emails_{opponent.replace('@', '_').replace('.', '_')}.txt"
                try:
                    with open(txt_file_path, 'w', encoding='utf-8') as f:
                        for email in structurally_cleaned_emails:
                            
                            f.write(f"ID: {email['문서No']}\n") 
                            f.write(f"보낸사람: {email['sender']}\n")
                            f.write(f"받은사람: {email['receiver']}\n")
                            f.write(f"날짜: {email['date']}\n")
                            f.write(f"제목: {email['subject']}\n")
                            #f.write(f"내용:\n{email.get('content', '')}\n\n")
                            # Write the final cleaned content (potentially pattern-removed)
                            f.write(f"내용:\n{email.get('cleaned_content', '')}\n\n")
                            f.write("---End---\n\n ")
                            
                    logging.info(f"상대방 {opponent} - TXT 파일 생성 완료: {txt_file_path}")
                    print(f"상대방 {opponent} - TXT 파일 생성됨: {txt_file_path}")
                except Exception as e:
                    logging.error(f"상대방 {opponent} - TXT 파일 생성 중 오류 발생: {e}")
                    print(f"상대방 {opponent} - TXT 파일 생성 중 오류 발생: {e}")
            else:
                logging.info(f"상대방 {opponent} - 처리할 이메일 없음. TXT 파일 생성 건너뜀.")

            # Store the final processed emails in the shared store
            grouped_emails[opponent] = structurally_cleaned_emails
            logging.info(f"상대방 {opponent} 처리 완료")
            print(f"상대방 {opponent} 처리 완료")

        logging.info(f"그룹화 완료 - 처리된 상대방 수: {len(grouped_emails)}")
        print(f"그룹화 완료 - 처리된 상대방 수: {len(grouped_emails)}")
        shared_store['grouped_emails'] = grouped_emails
        logging.info("CleanAndGroupEmailsNode 실행 완료")
        print("CleanAndGroupEmailsNode 실행 완료")
        return shared_store
    
    def clean_email_content(self, content):
        """
        이메일 내용에서 불필요한 HTML 태그를 제거하고 텍스트와 테이블 내용을 추출합니다.
        테이블은 지정된 형식으로 변환 후 삽입됩니다.

        Args:
            content (str): 이메일 본문 내용

        Returns:
            str: 정제된 이메일 내용
        """
        if not content:
            logging.info("clean_email_content: Input content is empty. Returning empty string.") # Changed to info
            return ""
        
        logging.info(f"clean_email_content: START - Input content length: {len(content)}") # Changed to info

        try:
            # --- 0. 초기 파싱 --- 
            soup = BeautifulSoup(content, 'html.parser')
            logging.info(f"clean_email_content: Parsed with BeautifulSoup. Soup object type: {type(soup)}") # Changed to info

            # --- 1. 전처리: <br> 태그를 임시 문자열로 교체 ---
            for br in soup.find_all('br'):
                br.replace_with('___BR___')

            # --- 2. 불필요한 태그 및 속성 제거 ---
            for tag_name in ['blockquote', 'style', 'script', 'meta', 'head', 
                             'o:p', 'o:shapedefaults', 'o:shapelayout', 
                             'v:shape', 'v:shapetype', 'v:imagedata', 'v:textbox', 'v:fill', 'v:stroke', 'v:shadow', 'v:path', 
                             'm:math', 'w:worddocument', 'xml', 'st1:*']:
                # 네임스페이스 포함 태그 제거 위해 find_all 사용
                tags_to_remove = soup.find_all(tag_name)
                for tag in tags_to_remove:
                     if tag:
                         tag.decompose()

            # 인라인 스타일 및 불필요한 속성 제거
            for tag in soup.find_all(True):
                # MS Office 네임스페이스 속성 제거 (예: xmlns:v, xmlns:o 등)
                attrs_to_remove = [attr for attr in tag.attrs if ':' in attr and attr.startswith(('xmlns:', 'v:', 'o:', 'w:', 'm:', 'x:', 'st1:'))]
                # 일반적인 불필요 속성 추가
                attrs_to_remove.extend(['style', 'class', 'lang', 'align', 'valign', 'width', 'height', 'nowrap', 'border', 'cellspacing', 'cellpadding', 'face', 'size'])
                for attr in set(attrs_to_remove): # 중복 제거
                     if tag.has_attr(attr):
                         del tag[attr]
            logging.info(f"clean_email_content: After Step 2 (Tag/Attribute Removal) - soup text length approx: {len(soup.get_text())}")

            # --- 3. 테이블 처리 및 텍스트 추출 ---
            saved_tables = []
            tables = soup.find_all('table')
            for table in tables:
                table_lines = []
                table_lines.append("--- Table ---")
                rows = table.find_all('tr')
                for row in rows:
                    # 셀 텍스트 추출 (빈 셀은 공백 ' '으로 처리, separator=''로 셀 내 줄바꿈 무시)
                    cols = [cell.get_text(separator='', strip=True) or ' ' for cell in row.find_all(['td', 'th'])]
                    table_lines.append(' | '.join(cols))
                table_lines.append("-------------")
                formatted_table_string = '\n'.join(table_lines)

                # 2. 형식화된 테이블 저장
                saved_tables.append(formatted_table_string)

                # 테이블 태그 제거
                table.decompose()

            # Extract main text AFTER removing tables
            main_text = soup.get_text(separator=' ', strip=True) # Use space separator initially

            # Restore BR tags AFTER getting text
            main_text = main_text.replace('___BR___', '\n')

            # Create combined table string
            tables_combined_string = "\n\n".join(saved_tables)

            # Log after text and table extraction/processing
            logging.info(f"clean_email_content: Extracted main_text length (after BR restore): {len(main_text)}")
            logging.info(f"clean_email_content: Extracted tables_combined_string length: {len(tables_combined_string)}")

            # --- 4. 텍스트와 테이블 결합 --- 
            if main_text and tables_combined_string:
                cleaned_text = f"{main_text}\n\n{tables_combined_string}"
            elif main_text:
                cleaned_text = main_text
            elif tables_combined_string:
                cleaned_text = tables_combined_string
            else:
                cleaned_text = "" # 둘 다 비어있으면 빈 문자열
            logging.info(f"clean_email_content: After combining text and tables - cleaned_text length: {len(cleaned_text)}") # Changed to info

            # --- 후처리: 불필요한 빈 줄 및 공백 정리 ---
            if cleaned_text: # cleaned_text가 비어있지 않을 때만 처리
                lines = cleaned_text.split('\n')
                final_lines = []
                prev_line_empty = True # 시작을 빈 줄 앞에 있는 것으로 간주
                for line in lines:
                    stripped_line = line.strip() # 각 줄의 앞뒤 공백 제거
                    if stripped_line:
                        final_lines.append(stripped_line) # 내용이 있는 줄 추가
                        prev_line_empty = False
                    elif not prev_line_empty:
                        # 이전 줄에 내용이 있었고, 현재 줄이 비어있으면 -> 단일 빈 줄(단락 구분) 추가
                        final_lines.append('')
                        prev_line_empty = True
                # 마지막에 추가된 빈 줄 제거 (있을 경우)
                while final_lines and not final_lines[-1]:
                    final_lines.pop()

                cleaned_text = '\n'.join(final_lines)
            # --- 후처리 끝 ---
            logging.info(f"clean_email_content: After post-processing (blank lines/spaces) - cleaned_text length: {len(cleaned_text)}") # Changed to info

            # Remove content after "-----Original Message-----"
            original_message_marker = "-----Original Message-----"
            marker_index = cleaned_text.find(original_message_marker)
            if marker_index != -1:
                original_length = len(cleaned_text)
                cleaned_text = cleaned_text[:marker_index].strip()
                logging.info(f"clean_email_content: Removed 'Original Message' part. Length changed from {original_length} to {len(cleaned_text)}.") # Changed to info
            else:
                 logging.info("clean_email_content: 'Original Message' marker not found.") # Changed to info

            # 최종 결과 반환
            logging.info(f"clean_email_content: FINAL - Returning cleaned_text length: {len(cleaned_text)}") # Changed to info
            return cleaned_text

        # except 블록 (try 블록과 같은 레벨로 들여쓰기 수정)
        except Exception as e:
            logging.error(f"이메일 내용 정제 중 오류 발생: {e}")
            print(f"Error processing HTML in clean_email_content: {e}")
            # 오류 발생 시 원본 반환 (혹은 빈 문자열)
            return content # 또는 ""

if __name__ == "__main__":
    db_path = "d:\\PythonProject\\llm\\email_analyzer2\\data\\temp_Portable_search_DB_1062개.db"
    node = CleanAndGroupEmailsNode(db_path)
    shared_store = {'contact_frequencies': {'example@domain.com': 10}, 'owner_email': 'owner@example.com', 'opponent_frequencies': {'opponent@example.com': 5}}
    updated_store = node.process(shared_store)
    print("상대방별 그룹화된 이메일 수:")
    for opponent, emails in updated_store['grouped_emails'].items():
        print(f"상대방: {opponent}, 이메일 수: {len(emails)}")
