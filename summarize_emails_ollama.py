import ollama
import glob
import os
import yaml

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

def summarize_email_file(client, model_name, prompt_template, file_path):
    """주어진 텍스트 파일 내용을 Ollama를 사용하여 요약합니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            email_content = f.read()

        if not email_content.strip():
            print(f"파일 '{os.path.basename(file_path)}' 내용이 비어있어 건너뜁니다.")
            return None

        if isinstance(prompt_template, str):
            prompt = prompt_template.format(email_content)
        else:
            print("오류: 프롬프트 템플릿이 문자열 형식이 아닙니다.")
            return None

        print(f"\n--- '{os.path.basename(file_path)}' 요약 중... ---")
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
        print(f"파일 '{os.path.basename(file_path)}' 요약 중 오류 발생: {e}")
        return None

def main():
    data_folder = 'data'
    prompt_file = 'prompts/email_summary_prompt.yaml'
    model_name = 'qwen3:4b'

    prompt_template = load_prompt(prompt_file)
    if not prompt_template:
        return

    email_files = glob.glob(os.path.join(data_folder, 'emails_*.txt'))

    if not email_files:
        print(f"'{data_folder}' 폴더에서 'emails_*.txt' 파일을 찾을 수 없습니다.")
        return

    try:
        client = ollama.Client()
    except Exception as e:
        print(f"Ollama 클라이언트 초기화 중 오류 발생: {e}")
        print("Ollama 서버가 실행 중인지 확인하세요.")
        return

    for file_path in email_files:
        summarize_email_file(client, model_name, prompt_template, file_path)

if __name__ == "__main__":
    main()
