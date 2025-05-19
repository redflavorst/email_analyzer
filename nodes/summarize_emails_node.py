from dotenv import load_dotenv
import os
import yaml
import logging
import requests
import json
import re
import tempfile
import uuid
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional
from utils.load_emails_for_contact import extract_email
import multiprocessing
from functools import partial

import time


load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SummarizeEmailsNode:
    """
    Ollama 또는 gpt-4.1 mini 등 다양한 요약 모델을 사용하여 이메일 데이터를 2단계로 요약하고 파일로 관리하는 노드:
    1. 개별 이메일 요약을 임시 파일에 저장
    2. 임시 파일을 읽어 전체 요약 생성
    """
    def __init__(self, model_provider="ollama", model_name="qwen3:4b", output_dir="data/summaries", ollama_base_url="http://localhost:11434", openai_api_key=None):
        self.single_prompt_path = 'd:\\PythonProject\\llm\\email_analyzer2\\prompts\\single_email_summary_prompt.yaml'
        self.total_prompt_path = 'd:\\PythonProject\\llm\\email_analyzer2\\prompts\\total_email_summary_prompt.yaml'
        self.single_prompt_template = self._load_prompt_template(self.single_prompt_path)
        self.total_prompt_template = self._load_prompt_template(self.total_prompt_path)
        self.model_provider = model_provider # 'ollama' 또는 'gpt4mini' 등
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url
        self.openai_api_key = openai_api_key
        self.temp_dir = tempfile.gettempdir()
        # LLM 파라미터 초기화
        self.max_tokens = 2000
        self.temperature = 0.3
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        logging.info(f"SummarizeEmailsNode: Output directory set to {self.output_dir}")

        # gpt-4.1 mini용 유틸리티 임포트 (필요시)
        try:
            from utils.call_gpt4mini import call_gpt4mini_api
            self.call_gpt4mini_api = call_gpt4mini_api
        except ImportError:
            self.call_gpt4mini_api = None
            logging.warning("gpt-4.1 mini 호출 유틸리티를 불러올 수 없습니다.")


    def _load_prompt_template(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                return data.get('prompt_template', '')
        except FileNotFoundError:
            logging.error(f"프롬프트 파일을 찾을 수 없습니다: {file_path}")
            return ""
        except Exception as e:
            logging.error(f"프롬프트 파일 로드 중 오류 발생 ({file_path}): {e}")
            return ""

    def _call_llm_api(self, prompt, max_tokens=2000, temperature=0.3):
        """
        provider/model_name에 따라 ollama 또는 gpt-4.1 mini 등 다양한 LLM API를 호출합니다.
        """
        # Ollama 모델 호출
        if self.model_provider == "ollama":
            api_url = f"{self.ollama_base_url}/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            }
            start_time = time.time()
            log_dir = os.path.join(self.output_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            api_log_file_path = os.path.join(log_dir, "api_call_times.txt")
            try:
                response = requests.post(api_url, json=payload)
                response.raise_for_status()
                response_data = response.json()
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000
                pid = os.getpid()
                log_message = f"PID: {pid} - Ollama LLM API call to '{self.model_name}' duration: {duration_ms:.2f} ms"
                logging.info(log_message)
                with open(api_log_file_path, 'a', encoding='utf-8') as f_log:
                    f_log.write(f"{datetime.now().isoformat()} - {log_message}\n")
                if response_data and 'response' in response_data and response_data['response']:
                    raw_response_text = response_data['response']
                    processed_text = re.sub(r"<think>.*?</think>", "", raw_response_text, flags=re.DOTALL).strip()
                    if "<think>" in processed_text:
                        if processed_text.startswith("<think>"):
                            original_parts = raw_response_text.split("<think>", 1)
                            if len(original_parts) > 0 and original_parts[0].strip():
                                processed_text = original_parts[0].strip()
                            else:
                                processed_text = "요약 생성 중 오류: LLM 응답이 <think> 태그로 시작하고 종료되지 않음"
                        else:
                            parts = processed_text.split("<think>", 1)
                            processed_text = parts[0].strip()
                    logging.info(f"Ollama LLM API 응답 수신 (길이: {len(processed_text)}) - '{processed_text[:50]}{'...' if len(processed_text) > 50 else ''}'")
                    return processed_text
                else:
                    end_time_error = time.time()
                    duration_ms_error = (end_time_error - start_time) * 1000
                    pid_error = os.getpid()
                    error_log_message = f"PID: {pid_error} - Ollama LLM API call to '{self.model_name}' resulted in empty or missing response. Duration: {duration_ms_error:.2f} ms. Response: {response_data}"
                    logging.warning(error_log_message)
                    with open(api_log_file_path, 'a', encoding='utf-8') as f_log:
                        f_log.write(f"{datetime.now().isoformat()} - {error_log_message}\n")
                    return "Error: LLM response was empty or not found in expected format."
            except Exception as e:
                logging.error(f"Ollama LLM API 호출 오류: {e}")
                return f"Error: Ollama API 호출 실패 - {e}"
        # gpt-4.1 mini(OpenAI 호환) 호출
        elif self.model_provider == "gpt4mini":
            if self.call_gpt4mini_api is None:
                logging.error("gpt-4.1 mini 호출 유틸리티가 없습니다. utils/call_gpt4mini.py를 확인하세요.")
                return "Error: gpt-4.1 mini 호출 유틸리티 없음"
            try:
                result = self.call_gpt4mini_api(prompt, api_key=self.openai_api_key, model=self.model_name, max_tokens=max_tokens, temperature=temperature)
                logging.info(f"gpt-4.1 mini API 응답 수신 (길이: {len(result) if result else 0}) - '{result[:50]}{'...' if result and len(result) > 50 else ''}'")
                return result
            except Exception as e:
                logging.error(f"gpt-4.1 mini API 호출 오류: {e}")
                return f"Error: gpt-4.1 mini API 호출 실패 - {e}"
        else:
            logging.error(f"지원하지 않는 model_provider: {self.model_provider}")
            return f"Error: 지원하지 않는 model_provider: {self.model_provider}"

    def _process_single_email(self, email: Dict[str, Any], idx: int, contact_email: str) -> Dict[str, Any]:
        """단일 이메일을 처리하여 요약 데이터를 반환 (병렬 프로세스에서 호출)."""
        current_process_id = os.getpid()
        logging.info(f"PID: {current_process_id} - 처리 시작: 이메일 idx {idx} (연락처: {contact_email})")
        original_email_id = email.get('id', email.get('문서No', f'fallback_id_{idx}'))
        content_for_prompt = email.get('cleaned_content', '')
        raw_sender = email.get('sender', '')
        raw_receiver = email.get('receiver', '')
        extracted_sender_email = extract_email(raw_sender) if raw_sender else 'N/A'
        extracted_receiver_email = extract_email(raw_receiver) if raw_receiver else 'N/A'
        if not extracted_sender_email and raw_sender:
            extracted_sender_email = 'N/A'
        if not extracted_receiver_email and raw_receiver:
            extracted_receiver_email = 'N/A'
        sender_for_prompt = extracted_sender_email if extracted_sender_email else 'N/A'
        receiver_for_prompt = extracted_receiver_email if extracted_receiver_email else 'N/A'
        summary_data = {
            'idx': idx + 1,
            'date': email.get('date', 'N/A'),
            'sender': sender_for_prompt,
            'receiver': receiver_for_prompt,
            'subject': email.get('subject', 'N/A'),
            'summary': 'Error: Processing failed before summarization'
        }
        try:
            logging.debug(f"Processing email {summary_data['idx']} for {contact_email} (Original ID: {original_email_id}) in process {os.getpid()})")
            prompt = self.single_prompt_template.format(
                sender=summary_data['sender'],
                receiver=summary_data['receiver'],
                date=summary_data['date'],
                subject=summary_data['subject'],
                content=content_for_prompt
            )
            # 모델 파라미터를 self에서 가져오도록 수정
            llm_summary = self._call_llm_api(prompt, max_tokens=self.max_tokens, temperature=self.temperature)
            if not llm_summary.startswith("Error"):
                summary_data['summary'] = llm_summary
            else:
                summary_data['summary'] = f"{llm_summary}"
                logging.warning(f"개별 요약 생성 실패 (idx: {summary_data['idx']}, Original ID: {original_email_id}): {llm_summary}")
        except Exception as e:
            error_message = f"Error during processing email idx {summary_data['idx']}: {type(e).__name__}: {e}"
            summary_data['summary'] = error_message
            logging.error(f"PID: {current_process_id} - 오류: 이메일 idx {summary_data['idx']} (연락처: {contact_email}, 원본 ID: {original_email_id}): {e}", exc_info=True)
        logging.info(f"PID: {current_process_id} - 처리 완료: 이메일 idx {idx} (연락처: {contact_email})")
        return summary_data

    def summarize_individual_emails(self, contact_email: str, emails: List[Dict[str, Any]]) -> tuple[str, list]:
        logging.info(f"DEBUG: summarize_individual_emails 진입 - 연락처: {contact_email}, 이메일 수: {len(emails)}")
        if not emails:
            logging.warning(f"요약할 이메일이 없습니다 ({contact_email})")
            return "", []

        # Sanitize contact_email for filename
        sanitized_contact_email = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', contact_email)
        csv_filename = f"{sanitized_contact_email}_communication_summary.csv"
        result_csv_file_path = os.path.join(self.output_dir, csv_filename)

        individual_summaries_list = []
        fieldnames = ['idx', 'date', 'sender', 'receiver', 'subject', 'summary']
        success_count = 0

        try:
            # 병렬 처리 설정
            # Use a smaller number of processes if CPU cores are few, or Ollama itself is a bottleneck
            num_processes = min(max(1, multiprocessing.cpu_count() // 2), 6) # Try half the cores, min 1, max 6
            # num_processes = 1 # For debugging single process flow
            logging.info(f"병렬 처리 시작 - 연락처: {contact_email}, 사용할 프로세스 수: {num_processes}, (CPU count: {multiprocessing.cpu_count()}), 처리할 이메일 수: {len(emails)}")

            # Prepare arguments for starmap: list of (email_dict, index) tuples
            # Each tuple in this list will be unpacked as arguments to _process_single_email (after self and contact_email)
            email_processing_args_for_starmap = [(email, i) for i, email in enumerate(emails)]


            
            # Use partial to pass 'self' (implicitly) and 'contact_email' to the worker function
            # _process_single_email expects (self, email, idx, contact_email)
            process_func_with_contact = partial(self._process_single_email, contact_email=contact_email)

            # --- 조건부 처리 로깅 추가 시작 ---
            is_sequential_condition_met = len(emails) < num_processes and len(emails) < 5
            logging.info(f"Contact: {contact_email} - 병렬/순차 처리 조건 확인: num_emails={len(emails)}, num_processes={num_processes}. "
                         f"순차 처리 조건 (len(emails) < num_processes AND len(emails) < 5) 결과: {is_sequential_condition_met}")
            # --- 조건부 처리 로깅 추가 끝 ---

            if is_sequential_condition_met: # 기존: len(emails) < num_processes and len(emails) < 5
                 logging.info(f"Contact: {contact_email} - 이메일 수가 적어 ({len(emails)}개) 순차 처리합니다.")
                 results = [process_func_with_contact(email_arg_tuple[0], email_arg_tuple[1]) for email_arg_tuple in email_processing_args_for_starmap]
            else:
                logging.info(f"Contact: {contact_email} - 이메일 수가 {len(emails)}개 이므로 병렬 처리합니다 with {num_processes} processes.") # << 추가된 로그
                with multiprocessing.Pool(processes=num_processes) as pool:
                    results = pool.starmap(process_func_with_contact, email_processing_args_for_starmap)
            
            
            # 결과 수집 및 CSV 쓰기
            with open(result_csv_file_path, 'w', newline='', encoding='utf-8-sig') as result_csvfile:
                writer = csv.DictWriter(result_csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for summary_data in results:
                    if summary_data: # Ensure summary_data is not None if a process failed unexpectedly before returning
                        writer.writerow(summary_data)
                        individual_summaries_list.append(summary_data)
                        if not summary_data['summary'].startswith("Error"):
                            success_count += 1
                    else:
                        logging.error(f"A process returned None for contact {contact_email}. This should not happen.")

            logging.info(f"병렬 처리 완료 - CSV 저장 경로: {result_csv_file_path}, 성공 요약 수: {success_count}/{len(emails)} for {contact_email}")

        except IOError as e:
            logging.error(f"결과 CSV 파일 생성/쓰기 오류 ({result_csv_file_path}): {e}")
            return "", []
        except Exception as e:
            logging.error(f"개별 요약 병렬 프로세스 중 오류 발생 ({contact_email}): {e}", exc_info=True)
            return "", []

        logging.info(f"DEBUG: summarize_individual_emails 종료 - CSV 저장 경로: {result_csv_file_path}, 생성된 요약 수: {len(individual_summaries_list)}")
        return result_csv_file_path, individual_summaries_list

    def summarize_overall(self, contact_email: str, individual_summaries: List[Dict[str, Any]]) -> str:
        logging.info(f"DEBUG: summarize_overall 진입 - 연락처: {contact_email}, 개별 요약 수: {len(individual_summaries)}")
        if not individual_summaries:
            logging.warning(f"개별 요약이 없어 전체 요약을 생성할 수 없습니다 ({contact_email})")
            return "개별 요약 정보가 없어 전체 요약을 생성할 수 없습니다."
        formatted_summaries = "\n".join([
            f"- {s['date'][:10]} ({s['sender']} → {s['receiver']}): {s['summary']}" 
            for s in individual_summaries
        ])
        prompt_data = {
            "contact_email": contact_email,
            "num_emails": len(individual_summaries),
            "individual_summaries": formatted_summaries
        }
        formatted_prompt = self.total_prompt_template.format(**prompt_data)
        # 전체 요약도 선택된 모델로 호출
        overall_summary = self._call_llm_api(formatted_prompt, max_tokens=20000, temperature=0.3)
        logging.info(f"DEBUG: summarize_overall 종료 - 연락처: {contact_email}, 전체 요약 길이: {len(overall_summary)}")
        return overall_summary

    def process(self, shared_store: Dict[str, Any]) -> Dict[str, Any]:
        logging.info("SummarizeEmailsNode 실행 시작...")
        if 'grouped_emails' not in shared_store or not shared_store['grouped_emails']:
            logging.warning("grouped_emails를 찾을 수 없거나 비어있습니다. SummarizeEmailsNode 처리를 건너뜁니다.")
            shared_store['email_summaries'] = {}
            return shared_store
        email_summaries_store = {}
        grouped_emails = shared_store['grouped_emails']
        num_contacts = len(grouped_emails)
        for i, (contact_email, emails) in enumerate(grouped_emails.items()):
            logging.info(f"연락처 {contact_email} ({i+1}/{num_contacts}) 처리 중...")
            csv_file_path, individual_summaries_list = self.summarize_individual_emails(contact_email, emails)
            overall_summary_text = ""
            if individual_summaries_list:
                logging.info(f"DEBUG: process 메서드 - summarize_overall 호출 직전. 연락처: {contact_email}, 개별 요약 수: {len(individual_summaries_list)}")
                overall_summary_text = self.summarize_overall(contact_email, individual_summaries_list)
                logging.info(f"DEBUG: process 메서드 - summarize_overall 호출 완료. 결과 길이: {len(overall_summary_text) if overall_summary_text else 0}")
                sanitized_contact_email = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', contact_email)
                overall_summary_filename = f"{sanitized_contact_email}_overall_summary.md"
                overall_summary_filepath = os.path.join(self.output_dir, overall_summary_filename)
                try:
                    with open(overall_summary_filepath, 'w', encoding='utf-8') as f:
                        f.write(f"# {contact_email} 종합 요약\n\n")
                        f.write(overall_summary_text)
                    logging.info(f"'{contact_email}'의 전체 요약을 '{overall_summary_filepath}'에 저장했습니다.")
                except Exception as e:
                    logging.error(f"'{overall_summary_filepath}'에 전체 요약을 저장하는 중 오류 발생: {e}")
            else:
                logging.warning(f"'{contact_email}'에 대한 개별 요약이 없어 전체 요약을 생성하거나 저장하지 않습니다.")
            email_summaries_store[contact_email] = {
                'individual_csv_path': csv_file_path,
                'individual': individual_summaries_list,
                'overall': overall_summary_text
            }
            logging.info(f"연락처 {contact_email} ({i+1}/{num_contacts}): 전체 요약 처리 완료.")
        shared_store['email_summaries'] = email_summaries_store
        logging.info("SummarizeEmailsNode 실행 완료.")
        return shared_store


if __name__ == "__main__":
    test_emails = [
        {
            'id': 'test001',
            'sender': 'pingo@example.com',
            'receiver': 'rex@example.com',
            'date': '2017-10-16',
            'subject': 'Important Opportunity',
            'cleaned_content': 'Hi Rex, Just wanted to mention a significant opportunity for 2018: the best TT rider will be using HJC helmets. Let\'s discuss.'
        },
        {
            'id': 'test002',
            'sender': 'pingo@example.com',
            'receiver': 'rex@example.com; team@example.com',
            'date': '2017-10-23',
            'subject': 'News: Wiggle acquired Bike24',
            'cleaned_content': 'FYI - Wiggle acquired Bike24 for 100 million euros. Maybe related to Brexit?'
        }
    ]
    shared_store_test = {'grouped_emails': {'rex@example.com': test_emails}}

    print("Ollama를 사용한 요약 테스트 시작...")
    node = SummarizeEmailsNode()

    try:
        requests.get(node.ollama_base_url, timeout=2)
        print(f"Ollama 엔드포인트 ({node.ollama_base_url}) 연결 가능. 모델 '{node.ollama_model}'을 사용합니다.")

        updated_store = node.process(shared_store_test)

        print("\n--- 요약 결과 ---")
        if 'email_summaries' in updated_store:
            for contact, overall_summary in updated_store['email_summaries'].items():
                print(f"\n## 연락처: {contact}")
                print("### 종합 요약:")
                print(overall_summary)
        else:
            print("요약 결과가 없습니다.")

    except requests.exceptions.ConnectionError:
        print(f"오류: Ollama 엔드포인트({node.ollama_base_url})에 연결할 수 없습니다. Ollama가 실행 중인지 확인하세요.")
    except Exception as e:
        print(f"테스트 중 오류 발생: {e}")