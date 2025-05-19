import ollama
import glob
import os
import yaml
import json
from collections import defaultdict
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from load_emails_for_contact import load_emails_for_contact, extract_email

def load_prompt(prompt_path):
    """YAML 프롬프트 파일 내용을 읽어 반환합니다."""
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('prompt_template')
    except FileNotFoundError:
        print(f"오류: 프롬프트 파일 '{prompt_path}'를 찾을 수 없습니다.")
        return None
    except yaml.YAMLError as e:
        print(f"오류: 프롬프트 파일 '{prompt_path}' 파싱 중 오류 발생: {e}")
        return None

def summarize_email_file(client, model_name, prompt_template, file_path=None, email=None):
    """주어진 텍스트 파일 또는 이메일 dict 내용을 Ollama를 사용하여 요약합니다."""
    try:
        if email is not None:
            email_content = email.get('body', email.get('content', ''))
            file_label = email.get('id', 'dict_email')
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                email_content = f.read()
            file_label = os.path.basename(file_path)

        if not email_content.strip():
            print(f"내용이 비어있어 건너뜁니다. (ID: {file_label})")
            return None

        if isinstance(prompt_template, str):
            prompt = prompt_template.format(email_content)
        else:
            print("오류: 프롬프트 템플릿이 문자열 형식이 아닙니다.")
            return None

        print(f"\n--- '{file_label}' 요약 중... ---")
        response = client.chat(model=model_name, messages=[
            {
                'role': 'user',
                'content': prompt,
            }
        ])
        summary = response['message']['content']
        print("요약 결과:")
        print(summary)
        return summary

    except FileNotFoundError:
        print(f"오류: 파일 '{file_path}'를 찾을 수 없습니다.")
        return None
    except Exception as e:
        print(f"파일 '{file_label}' 요약 중 오류 발생: {e}")
        return None

def main():
    # 사용자에게 DB 파일 경로 직접 입력받기
    db_path = input('이메일 SQLite DB 파일 경로를 입력하세요 (예: data/temp_Portable_search_DB_1062개.db): ').strip()
    if not db_path or not os.path.isfile(db_path):
        print('DB 파일 경로가 올바르지 않거나 파일이 존재하지 않습니다.')
        return
    prompt_file = 'prompts/email_summary_prompt.yaml'
    model_name = 'qwen3:4b'

    prompt_template = load_prompt(prompt_file)
    if not prompt_template:
        return

    try:
        client = ollama.Client()
    except Exception as e:
        print(f"Ollama 클라이언트 초기화 중 오류 발생: {e}")
        print("Ollama 서버가 실행 중인지 확인하세요.")
        return

    print("DB에서 이메일 로드 중...")
    # 일단 owner_email에 아무 값이나 넣고 전체 이메일 로드
    emails = load_emails_for_contact(db_path, owner_email="%")
    if not emails:
        print("이메일 데이터가 없습니다.")
        return
    print(f"총 {len(emails)}건 이메일 로드 완료.")

    # 연락처별로 분류 및 본인 후보 집계
    contact_count = defaultdict(int)
    contact_emails = defaultdict(list)
    for email in emails:
        sender = extract_email(email['sender'])
        receiver = extract_email(email['receiver'])
        # 집계: 보낸사람/받는사람 모두 카운트
        contact_count[sender] += 1
        contact_count[receiver] += 1
        # 분류용: 보낸사람/받는사람 모두 분류
        contact_emails[sender].append(email)
        contact_emails[receiver].append(email)

    # 본인(owner_email) 자동 추출: 메일 수가 가장 많은 주소
    sorted_contacts = sorted(contact_count.items(), key=lambda x: x[1], reverse=True)
    owner_email = sorted_contacts[0][0]
    print(f"\n[자동 감지] 본인(owner_email): {owner_email}")

    # 본인을 제외한 연락처별로 이메일 재분류 (본인과 주고받은 메일만)
    other_contacts = [c for c, _ in sorted_contacts if c != owner_email]
    # top N만 추출
    try:
        top_n = int(input(f"요약할 연락처 수(top N, 본인 제외, 최대 {len(other_contacts)}): "))
    except Exception:
        top_n = 5
    top_n_contacts = other_contacts[:top_n]
    print(f"\n요약 대상 연락처 (top {top_n}): {top_n_contacts}")

    # 각 연락처별로 본인과 주고받은 메일만 추출
    output_base = os.path.join('output', 'contacts')
    os.makedirs(output_base, exist_ok=True)

    for contact in top_n_contacts:
        # 본인과 주고받은 메일만 필터
        filtered_emails = [e for e in contact_emails[contact] if owner_email in (extract_email(e['sender']), extract_email(e['receiver'])) and contact in (extract_email(e['sender']), extract_email(e['receiver']))]
        # 중복 제거 (문서No 기준)
        unique_emails = {e.get('문서No', e.get('id', '')): e for e in filtered_emails}.values()
        email_list = list(unique_emails)
        print(f"\n[연락처: {contact}] 이메일 {len(email_list)}건 요약 중...")
        # id, date, subject, body 표준화
        for email in email_list:
            email['contact'] = contact
            email['id'] = str(email.get('문서No', email.get('id', '')))
            email['date'] = email.get('date', email.get('받은시간', ''))
            email['subject'] = email.get('subject', email.get('제목', ''))
            email['body'] = email.get('content', email.get('본문', ''))
        # 1. 개별 메일 요약
        for email in email_list:
            summary = summarize_email_file(client, model_name, prompt_template, None, email=email)
            email['summary'] = summary or ''
        # emails.json 저장
        contact_folder = os.path.join(output_base, contact)
        os.makedirs(contact_folder, exist_ok=True)
        with open(os.path.join(contact_folder, "emails.json"), "w", encoding="utf-8") as f:
            json.dump(email_list, f, ensure_ascii=False, indent=2)
        # 2. 전체 요약 및 주제별 ref_emails 샘플 (실제론 LLM 활용)
        activities = []
        delivery_emails = [e['id'] for e in email_list if '납기' in e['subject']]
        contract_emails = [e['id'] for e in email_list if '계약' in e['subject']]
        if delivery_emails:
            activities.append({
                'topic': '납기 일정 협의',
                'summary': '납기 일정 관련 문의가 반복됨.',
                'ref_emails': delivery_emails
            })
        if contract_emails:
            activities.append({
                'topic': '계약 진행',
                'summary': '계약 관련 논의가 있었음.',
                'ref_emails': contract_emails
            })
        etc_emails = [e['id'] for e in email_list if e['id'] not in delivery_emails + contract_emails]
        if etc_emails:
            activities.append({
                'topic': '기타',
                'summary': '기타 문의가 있었음.',
                'ref_emails': etc_emails
            })
        summary_json = {
            'contact': contact,
            'relationship_summary': f"{contact}와 주요 업무 관련 대화가 오갔음.",
            'activities': activities
        }
        with open(os.path.join(contact_folder, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary_json, f, ensure_ascii=False, indent=2)
    print("\n연락처별 emails.json, summary.json 생성 완료!")

if __name__ == "__main__":
    main()
