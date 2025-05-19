import logging
import sys
from utils.load_contact_frequencies import load_contact_frequencies_with_owner, load_opponent_frequencies
from utils.load_emails_for_contact import load_emails_for_contact
from nodes.load_email_node import LoadEmailNode
from nodes.clean_and_group_emails_node import CleanAndGroupEmailsNode
from nodes.summarize_emails_node import SummarizeEmailsNode
from nodes.report_node import ReportNode

# --- Start of new logging configuration ---
# Clear any existing handlers on the root logger
# This is to ensure our configuration takes precedence if any library called basicConfig already
root_logger = logging.getLogger()
if root_logger.hasHandlers():
    root_logger.handlers.clear()

# Configure logging to use UTF-8 for console output if possible
# Default to DEBUG level, adjust as needed (e.g., logging.INFO)
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Log to standard output
    ]
)

# Try to set stdout and stderr encoding to UTF-8 (Python 3.7+)
# This helps if the console itself supports UTF-8 but Python defaults to another encoding
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        # Use __name__ for the logger name to avoid potential conflicts if this runs multiple times
        logging.getLogger(__name__).info("Successfully reconfigured sys.stdout to UTF-8.") 
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not reconfigure sys.stdout to UTF-8: {e}")

if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
        logging.getLogger(__name__).info("Successfully reconfigured sys.stderr to UTF-8.")
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not reconfigure sys.stderr to UTF-8: {e}")
# --- End of new logging configuration ---

# DB 경로 설정
db_path = "d:\\PythonProject\\llm\\email_analyzer2\\data\\temp_Portable_search_DB_1062개.db"
output_path = "d:\\PythonProject\\llm\\email_analyzer2\\data\\email_report.md"

# 워크플로우 실행
def run_workflow():
    shared_store = {}
    
    # 이메일 데이터 로드
    print("이메일 데이터 로드 중...")
    freqs, owner = load_contact_frequencies_with_owner(db_path)
    shared_store['contact_frequencies'] = freqs
    shared_store['owner_email'] = owner
    print(f"메일 주인: {owner}")
    print(f"로드된 연락처 빈도수: {len(freqs)}")
    for contact, freq in list(freqs.items())[:10]:  # 상위 10개만 출력
        print(f"연락처: {contact}, 빈도수: {freq}")
    
    opponent_freqs = load_opponent_frequencies(db_path, owner)
    shared_store['opponent_frequencies'] = opponent_freqs
    print(f"로드된 상대방 빈도수: {len(opponent_freqs)}")
    for opponent, freq in list(opponent_freqs.items())[:10]:  # 상위 10개만 출력
        print(f"상대방: {opponent}, 빈도수: {freq}")
    
    # 노드 초기화
    load_node = LoadEmailNode(db_path)
    clean_node = CleanAndGroupEmailsNode(db_path)
<<<<<<< HEAD
    # --- 요약 모델 선택 안내 및 입력 ---
    print("\n[요약 모델 선택]")
    print("1. qwen3.0 (Ollama, 무료/로컬)")
    print("2. gpt4.1 mini (OpenAI, 유료/클라우드)")
    model_choice = input("사용할 요약 모델을 선택하세요 (1 또는 2): ").strip()
    if model_choice == "2":
        openai_api_key = input("OpenAI API 키를 입력하세요: ").strip()
        summarize_node = SummarizeEmailsNode(
            model_provider="gpt4mini",
            model_name="gpt-4.1-mini",
            openai_api_key=openai_api_key
        )
        print("gpt-4.1 mini(OpenAI)로 요약을 진행합니다.")
    else:
        summarize_node = SummarizeEmailsNode(
            model_provider="ollama",
            model_name="qwen3:4b"
        )
        print("qwen3.0(Ollama)로 요약을 진행합니다.")
=======
    summarize_node = SummarizeEmailsNode()
>>>>>>> 4517ff7eb17d4e04168e7a40d48df5dd675da644
    report_node = ReportNode(output_path)
    
    print("워크플로우 실행 시작")
    # 노드 실행
    shared_store = load_node.process(shared_store)
    print("LoadEmailNode 실행 완료")
    try:
        shared_store = clean_node.process(shared_store)
        print("CleanAndGroupEmailsNode 실행 완료")
    except Exception as e:
        print(f"CleanAndGroupEmailsNode 실행 중 오류 발생: {e}")
    try:
        print("SummarizeEmailsNode 실행 시작...")
        shared_store = summarize_node.process(shared_store)
        print("SummarizeEmailsNode 실행 완료")
    except Exception as e:
        print(f"SummarizeEmailsNode 실행 중 오류 발생: {e}")
    try:
        print("ReportNode 실행 시작...")
        shared_store = report_node.process(shared_store)
        print("ReportNode 실행 완료")
    except Exception as e:
        print(f"ReportNode 실행 중 오류 발생: {e}")
    print("워크플로우 실행 완료")
    
    return shared_store

if __name__ == "__main__":
    run_workflow()
    print("이메일 분석이 완료되었습니다. 보고서를 확인하세요:", output_path)
