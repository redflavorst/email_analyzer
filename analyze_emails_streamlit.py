import streamlit as st
import os
from utils.load_contact_frequencies import load_contact_frequencies_with_owner
from utils.load_emails_for_contact import load_emails_for_contact
from nodes.summarize_emails_node import SummarizeEmailsNode
import pandas as pd

st.set_page_config(page_title="이메일 요약 분석기", layout="wide")
st.title("📧 이메일 요약 분석기 (GPT-4.1 mini)")

# 1. 데이터베이스 파일 선택 (사이드바)
data_dir = os.path.join(os.path.dirname(__file__), "data")
db_files = [f for f in os.listdir(data_dir) if f.endswith(".db")]
db_file = st.sidebar.selectbox("분석할 DB 파일을 선택하세요", db_files)
db_path = os.path.join(data_dir, db_file)

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
    # 모델별 summarize_node 생성 분기
    if use_gpt4mini:
        openai_api_key = get_final_openai_key(openai_api_key_input)
        if not openai_api_key:
            st.error("OpenAI API Key를 입력하거나 .env에 등록해주세요!")
            st.stop()
        summarize_node = SummarizeEmailsNode(
            model_provider="gpt4mini",
            model_name="gpt-4.1-mini",
            openai_api_key=openai_api_key,
            output_dir=os.path.join(data_dir, "summaries_streamlit")
        )
    else:
        summarize_node = SummarizeEmailsNode(
            model_provider="ollama",
            model_name="qwen3:4b",
            output_dir=os.path.join(data_dir, "summaries_streamlit")
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
    for contact, _ in top_contacts:
        emails = load_emails_for_contact(db_path, contact)
        if not emails:
            summary_text = "이메일 데이터 없음"
        else:
            # 개별 요약 + 전체 요약
            _, individual_summaries = summarize_node.summarize_individual_emails(contact, emails)
            summary_text = summarize_node.summarize_overall(contact, individual_summaries)
        summary_results.append({
            "연락처": contact,
            "요약": summary_text
        })
    # 6. 결과 출력 (본문)
    st.write("## Top N 연락처별 이메일 요약 결과")
    for item in summary_results:
        st.markdown(f"### {item['연락처']}")
        summary = item['요약']
        if not summary or summary.strip() == "" or summary is None:
            st.info("요약 결과가 없습니다.")
        elif summary.lower().startswith("error") or "api 호출 실패" in summary:
            st.error(f"요약 실패: {summary}")
        else:
            # 길거나 포맷이 깨지는 경우 코드블록으로도 제공
            st.markdown(summary)
            st.code(summary, language="markdown")
