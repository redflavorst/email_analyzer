import os
import requests
import logging

def call_gpt4mini_api(prompt, api_key=None, model="gpt-4.1-mini", max_tokens=2000, temperature=0.3):
    """
    gpt-4.1 mini(OpenAI 호환) API 호출 함수
    :param prompt: 프롬프트 문자열
    :param api_key: OpenAI API Key (환경변수에서 불러올 수도 있음)
    :param model: 사용할 모델명
    :param max_tokens: 생성할 최대 토큰 수
    :param temperature: 생성 다양성
    :return: 요약 문자열
    """
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY", None)
    if api_key is None:
        raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다.")

    api_url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        logging.error(f"gpt-4.1 mini API 호출 오류: {e}")
        return None

if __name__ == "__main__":
    prompt = "이메일 내용을 요약해줘: 안녕하세요, 테스트 이메일입니다."
    print(call_gpt4mini_api(prompt, api_key="sk-..."))
