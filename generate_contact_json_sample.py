import os
import json
from collections import defaultdict

# 샘플 이메일 데이터 (실제 환경에서는 파일에서 읽어옴)
sample_emails = [
    {
        "id": "email_001",
        "date": "2023-04-01",
        "subject": "납기 일정 문의",
        "body": "4월 납기 일정 문의 및 확인 요청",
        "contact": "kim@company.com"
    },
    {
        "id": "email_002",
        "date": "2023-04-05",
        "subject": "계약서 첨부",
        "body": "계약서 첨부 및 서명 요청",
        "contact": "kim@company.com"
    },
    {
        "id": "email_003",
        "date": "2023-04-07",
        "subject": "납기 일정 재확인",
        "body": "납기 일정 재확인 요청",
        "contact": "kim@company.com"
    },
    {
        "id": "email_004",
        "date": "2023-04-02",
        "subject": "제품 문의",
        "body": "신제품 견적 문의",
        "contact": "lee@other.com"
    }
]

def simple_email_summarize(email):
    # 실제로는 LLM 사용, 여기선 본문 앞 10자만
    return email["body"][:10] + "...요약"

def generate_contact_jsons(emails, output_dir):
    contacts = defaultdict(list)
    for email in emails:
        contacts[email["contact"]].append(email)

    for contact, email_list in contacts.items():
        # 1. 개별 메일 요약
        for email in email_list:
            email["summary"] = simple_email_summarize(email)
        
        # emails.json 저장
        contact_folder = os.path.join(output_dir, contact)
        os.makedirs(contact_folder, exist_ok=True)
        with open(os.path.join(contact_folder, "emails.json"), "w", encoding="utf-8") as f:
            json.dump(email_list, f, ensure_ascii=False, indent=2)

        # 2. 전체 요약 및 주제별 ref_emails 샘플 (실제론 LLM 활용)
        activities = []
        # 샘플: 납기 관련, 계약 관련으로 분류
        delivery_emails = [e["id"] for e in email_list if "납기" in e["subject"]]
        contract_emails = [e["id"] for e in email_list if "계약" in e["subject"]]
        if delivery_emails:
            activities.append({
                "topic": "납기 일정 협의",
                "summary": "납기 일정 관련 문의가 반복됨.",
                "ref_emails": delivery_emails
            })
        if contract_emails:
            activities.append({
                "topic": "계약 진행",
                "summary": "계약 관련 논의가 있었음.",
                "ref_emails": contract_emails
            })
        # 기타
        etc_emails = [e["id"] for e in email_list if e["id"] not in delivery_emails + contract_emails]
        if etc_emails:
            activities.append({
                "topic": "기타",
                "summary": "기타 문의가 있었음.",
                "ref_emails": etc_emails
            })
        summary_json = {
            "contact": contact,
            "relationship_summary": f"{contact}와 주요 업무 관련 대화가 오갔음.",
            "activities": activities
        }
        with open(os.path.join(contact_folder, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary_json, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 실제 환경에서는 실제 이메일 파싱 결과를 사용
    generate_contact_jsons(sample_emails, output_dir="output/contacts")
    print("샘플 emails.json, summary.json 파일이 생성되었습니다.")
