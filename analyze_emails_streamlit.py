import streamlit as st
import os
from utils.load_contact_frequencies import load_contact_frequencies_with_owner
from utils.load_emails_for_contact import load_emails_for_contact
from nodes.summarize_emails_node import SummarizeEmailsNode
import pandas as pd

st.set_page_config(page_title="이메일 요약 분석기", layout="wide")
st.title("📧 이메일 요약 분석기 (GPT-4.1 mini)")

# 1. 데이터베이스 파일 선택 (사이드바)
db_file = st.sidebar.file_uploader("SQLite DB 파일을 업로드하세요", type=["db"])
if db_file is not None:
    temp_db_path = "temp_uploaded_db.db"
    with open(temp_db_path, "wb") as f:
        f.write(db_file.getbuffer())
    db_path = temp_db_path
    st.success(f"DB 파일 업로드 완료: {db_file.name}")
else:
    st.info("DB 파일을 먼저 업로드 해주세요.")
    st.stop()

# 2. 분석 옵션 (Top N)
top_n = st.sidebar.number_input("가장 많이 주고받은 사용자 Top N", min_value=1, max_value=10, value=3, step=1)

# 3. LLM 모델 선택 및 OpenAI API 키 입력 (사이드바)
from dotenv import load_dotenv
load_dotenv()
def get_env_openai_key():
    import os
    return os.environ.get("OPENAI_API_KEY", "")
def get_final_openai_key(user_input):
    return user_input.strip() if user_input.strip() else get_env_openai_key()
llm_model = st.sidebar.selectbox("요약 LLM 모델 선택", ["qwen3.0 (Ollama, 무료/로컬)", "gpt-4.1 mini (OpenAI, 유료/클라우드)"])
use_gpt4mini = llm_model.startswith("gpt-4.1")
openai_api_key_input = ""
if use_gpt4mini:
    openai_api_key_input = st.sidebar.text_input("OpenAI API Key 입력 (미입력시 .env 값 사용)", type="password", value="")

if st.sidebar.button("분석 실행"):
    # OpenAI API Key 전달 경로 점검
    import os
    from dotenv import load_dotenv
    load_dotenv()
    env_api_key = os.environ.get("OPENAI_API_KEY", "")
    openai_api_key = get_final_openai_key(openai_api_key_input)
    # 마스킹 표시
    def mask_key(key):
        if not key or len(key) < 8:
            return key
        return key[:6] + "..." + key[-4:]
    st.info(f"[디버깅] .env API Key: {mask_key(env_api_key)} | 입력값: {mask_key(openai_api_key_input)} | 최종 사용: {mask_key(openai_api_key)}")
    print(f"[DEBUG] .env API Key: {env_api_key}")
    print(f"[DEBUG] 입력값: {openai_api_key_input}")
    print(f"[DEBUG] 최종 사용: {openai_api_key}")

    # 모델별 summarize_node 생성 분기
    if use_gpt4mini:
        if not openai_api_key:
            st.error("OpenAI API Key를 입력하거나 .env에 등록해주세요!")
            st.stop()
        summarize_node = SummarizeEmailsNode(
            model_provider="gpt4mini",
            model_name="gpt-4.1-mini",
            openai_api_key=openai_api_key,
            output_dir=os.path.join(os.path.dirname(__file__), "summaries_streamlit")
        )
    else:
        summarize_node = SummarizeEmailsNode(
            model_provider="ollama",
            model_name="qwen3:4b",
            output_dir=os.path.join(os.path.dirname(__file__), "summaries_streamlit")
        )
    # 4. DB에서 소유주/연락처별 빈도수 로드
    with st.spinner("DB에서 메일 소유주 및 연락처 빈도수 로드 중..."):
        freqs, owner = load_contact_frequencies_with_owner(db_path)
    st.success(f"메일 소유주: {owner}")
    # Top N 연락처 선정 (메일 소유주 본인은 제외)
    filtered_contacts = [(c, f) for c, f in freqs.items() if c != owner]
    top_contacts = sorted(filtered_contacts, key=lambda x: x[1], reverse=True)[:top_n]
    st.write(f"### 메일을 가장 많이 주고받은 Top {top_n} 연락처 (본인 제외)")
    st.table(pd.DataFrame(top_contacts, columns=["연락처", "메일 수"]))

    # 5. 각 연락처별 이메일 로드 및 요약
    # summarize_node는 이미 위에서 생성되었으므로 여기서 다시 생성할 필요가 없습니다.
    summary_results = []
    import json
    for contact, _ in top_contacts:
        emails = load_emails_for_contact(db_path, contact)
        output_dir = os.path.join("output", "contacts", contact)
        os.makedirs(output_dir, exist_ok=True)
        emails_json_path = os.path.join(output_dir, "emails.json")
        if not emails:
            summary_text = "이메일 데이터 없음"
            emails_to_save = []
            summary_json_path = None
        else:
            # 개별 요약 + 전체 요약
            try:
                _, individual_summaries = summarize_node.summarize_individual_emails(contact, emails)
                summary_text = summarize_node.summarize_overall(contact, individual_summaries)
                # 각 이메일에 summary 붙이기 (길이 맞추기)
                for i, email in enumerate(emails):
                    summary_val = ''
                    if i < len(individual_summaries) and 'summary' in individual_summaries[i]:
                        summary_val = individual_summaries[i]['summary']
                    email['summary'] = summary_val
                # 저장할 필드만 추출
                def filter_email(email):
                    return {
                        "id": email.get("문서No") or email.get("id"),
                        "date": email.get("date"),
                        "subject": email.get("subject"),
                        "contact": contact,
                        "summary": email.get("summary", "")
                    }
                emails_to_save = [filter_email(e) for e in emails]
                # 2단계 구조 기반 summary.json 생성
                import yaml, re, json as pyjson
                # 1단계: 전체/주제별 요약 추출
                step1_path = os.path.join(os.path.dirname(__file__), "prompts", "step1_total_email_summary_prompt.yaml")
                try:
                    with open(step1_path, "r", encoding="utf-8") as pf:
                        step1_yaml = yaml.safe_load(pf)
                        step1_template = step1_yaml.get("prompt_template", "")
                except Exception as e:
                    st.error(f"step1 프롬프트 로드 실패: {e}")
                    step1_template = ""
                indiv_lines = []
                for email in emails_to_save:
                    indiv_lines.append(f"- id: {email['id']}, date: {email['date']}, subject: {email['subject']}, summary: {email['summary']}")
                indiv_summary_text = "\n".join(indiv_lines)
                step1_prompt = step1_template.format(contact_email=contact, individual_summaries=indiv_summary_text)
                summary_json_path = os.path.join(output_dir, "summary.json")
                step1_result = None
                try:
                    llm_result1 = summarize_node._call_llm_api(step1_prompt, max_tokens=2000, temperature=0.3)
                    json_match = re.search(r'```json\s*(.*?)\s*```', llm_result1, re.DOTALL)
                    if json_match:
                        json_str1 = json_match.group(1)
                    else:
                        json_str1 = llm_result1.strip()
                    step1_result = pyjson.loads(json_str1)
                except Exception as e:
                    st.error(f"1단계 LLM 요약/파싱 실패: {e}\nLLM 응답: {llm_result1 if 'llm_result1' in locals() else ''}")
                    summary_json_path = None
                    raise
                # 2단계: 주제별 ref_emails 추출
                step2_path = os.path.join(os.path.dirname(__file__), "prompts", "step2_topic_ref_emails_prompt.yaml")
                try:
                    with open(step2_path, "r", encoding="utf-8") as pf:
                        step2_yaml = yaml.safe_load(pf)
                        step2_template = step2_yaml.get("prompt_template", "")
                except Exception as e:
                    st.error(f"step2 프롬프트 로드 실패: {e}")
                    step2_template = ""
                # 주제별 요약 리스트 생성
                topics_summaries = "\n".join([
                    f"- topic: {a['topic']}, summary: {a['summary']}" for a in step1_result.get('activities', [])
                ])
                # emails.json 전체 요약 리스트
                email_summaries = "\n".join([
                    f"- id: {e['id']}, date: {e['date']}, subject: {e['subject']}, summary: {e['summary']}" for e in emails_to_save
                ])
                step2_prompt = step2_template.format(
                    contact_email=contact,
                    topics_summaries=topics_summaries,
                    email_summaries=email_summaries
                )
                try:
                    llm_result2 = summarize_node._call_llm_api(step2_prompt, max_tokens=2000, temperature=0.3)
                    json_match2 = re.search(r'```json\s*(.*?)\s*```', llm_result2, re.DOTALL)
                    if json_match2:
                        json_str2 = json_match2.group(1)
                    else:
                        json_str2 = llm_result2.strip()
                    ref_emails_list = pyjson.loads(json_str2)
                except Exception as e:
                    st.error(f"2단계 LLM 주제별 ref_emails 추출/파싱 실패: {e}\nLLM 응답: {llm_result2 if 'llm_result2' in locals() else ''}")
                    summary_json_path = None
                    raise
                # 3단계: 최종 summary.json 합치기
                activities = step1_result.get('activities', [])
                # topic 기준으로 ref_emails 매칭
                topic2ref = {r['topic']: r['ref_emails'] for r in ref_emails_list}
                for a in activities:
                    a['ref_emails'] = topic2ref.get(a['topic'], [])
                summary_json = {
                    "contact": step1_result.get("contact", contact),
                    "relationship_summary": step1_result.get("relationship_summary", ""),
                    "activities": activities
                }
                with open(summary_json_path, "w", encoding="utf-8") as f:
                    pyjson.dump(summary_json, f, ensure_ascii=False, indent=2)

            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                st.error(f"[디버깅] gpt-4.1 mini API 호출 중 예외 발생!\n{str(e)}\n{tb}")
                print(f"[DEBUG] gpt-4.1 mini API 호출 중 예외: {e}\n{tb}")
                summary_text = f"Error: gpt-4.1 mini API 호출 실패 - {str(e)}"
                emails_to_save = []
                summary_json_path = None
        # emails.json 저장 (summary 포함, content 제외)
        import json
        with open(emails_json_path, "w", encoding="utf-8") as f:
            json.dump(emails_to_save, f, ensure_ascii=False, indent=2)
        summary_results.append({
            "연락처": contact,
            "요약": summary_text,
            "emails_json_path": emails_json_path,
            "summary_json_path": summary_json_path,
            "이메일_개수": len(emails)
        })
    # 6. 결과 출력 (본문)
    st.write("## Top N 연락처별 이메일 요약 결과")
    import json
    for item in summary_results:
        st.markdown(f"### {item['연락처']}")
        summary_json_path = item.get('summary_json_path')
        emails_json_path = item.get('emails_json_path')
        if not summary_json_path or not os.path.exists(summary_json_path):
            st.info("summary.json 파일이 존재하지 않습니다.")
            continue
        try:
            with open(summary_json_path, "r", encoding="utf-8") as f:
                summary_json = json.load(f)
        except Exception as e:
            st.error(f"summary.json 파일 로드 실패: {e}")
            continue
        # 근거 메일 제목 매핑용 emails.json 로드
        email_id2subject = {}
        if emails_json_path and os.path.exists(emails_json_path):
            try:
                with open(emails_json_path, "r", encoding="utf-8") as ef:
                    emails_list = json.load(ef)
                    email_id2subject = {e.get("id"): e.get("subject", "") for e in emails_list}
            except Exception as e:
                st.warning(f"emails.json 로드 실패: {e}")
        # 비즈니스 관계 요약
        relationship_summary = summary_json.get('relationship_summary', '')
        st.markdown(f"**비즈니스 관계 요약:** {relationship_summary}")
        # activities 마크다운+팝업 출력
        activities = summary_json.get('activities', [])
        if not activities:
            st.info("주제별 요약(activities) 정보가 없습니다.")
            continue
        st.markdown("**주요 활동 요약**:")
        for idx, a in enumerate(activities, 1):
            topic = a.get("topic", "")
            summary = a.get("summary", "")
            ref_ids = a.get("ref_emails", [])
            # 참고 아이콘
            icon = "<sup>🔍</sup>" if ref_ids else ""
            # 마크다운 번호매김
            st.markdown(f"{idx}. **{topic}**: {summary} {icon}", unsafe_allow_html=True)
            if ref_ids:
                with st.expander(f"🔍 근거 메일 보기 - {topic}"):
                    # 제목 테이블
                    subjects = [
                        {"메일ID": eid, "제목": email_id2subject.get(eid, "(제목 없음)")}
                        for eid in ref_ids
                    ]
                    if subjects:
                        st.table(subjects)
                    else:
                        st.info("근거 메일이 없습니다.")
