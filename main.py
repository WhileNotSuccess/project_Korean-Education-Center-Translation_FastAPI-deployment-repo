# FastAPI, Transformers, BeautifulSoup, Pydantic, CORS 등 필요한 라이브러리들을 임포트합니다.
from fastapi import FastAPI, Response, status                     # FastAPI 앱 생성 및 응답 관련 모듈
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast  # mBART 모델과 토크나이저 로딩
from bs4 import BeautifulSoup                                      # HTML 파싱을 위한 BeautifulSoup
from pydantic import BaseModel                                     # 데이터 유효성 검사용 Pydantic 모델
from fastapi.middleware.cors import CORSMiddleware                 # CORS 미들웨어 (다른 도메인 요청 허용)
import json                                                        # JSON 파일 읽기/쓰기 모듈
import os                                                          # 파일 및 경로 관련 모듈
import re                                                          # 정규 표현식 모듈
from difflib import SequenceMatcher                                # 문자열 유사도 비교를 위한 SequenceMatcher

# 클라이언트로부터 받을 요청 데이터 구조를 정의 (HTML의 제목과 내용을 포함)
class HtmlBody(BaseModel):
    title: str        # HTML 문서의 제목
    content: str      # HTML 문서의 실제 내용

# FastAPI 애플리케이션 인스턴스를 생성합니다.
app = FastAPI()

# CORS(교차 출처 리소스 공유)를 허용할 도메인 목록을 정의합니다.
origins = [
    'http://localhost:5173',
    'http://localhost:8080',
    'http://localhost',
    'https://localhost.com',
    'https://www.bapull.store',
    'https://www.voicevocab.store'
]

# CORS 미들웨어를 추가하여 정의한 도메인에서의 요청을 허용합니다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # 위에 정의된 도메인들만 허용
    allow_credentials=True,       # 자격 증명(쿠키, 인증 등) 포함 허용
    allow_methods=["*"],          # 모든 HTTP 메소드 허용 (GET, POST, etc.)
    allow_headers=["*"],          # 모든 HTTP 헤더 허용
)

# mBART 모델에서 사용하는 언어 코드 매핑 딕셔너리
language_codes = {
    "en": "en_XX",   # 영어
    "ja": "ja_XX",   # 일본어
    "ko": "ko_KR"    # 한국어
}

# 모델을 캐싱하기 위한 딕셔너리 (모델이 여러 번 로드되는 것을 방지)
loaded = {}

# 사용자 정의 용어 및 변형(term variants)들을 저장한 JSON 파일을 로드하는 함수
def load_custom_term_variants():
    # 현재 파일의 위치를 기준으로 JSON 파일 경로 설정
    path = os.path.join(os.path.dirname(__file__), "custom_terms_map.json")
    if os.path.exists(path):
        # JSON 파일이 존재하면 해당 파일을 열어 파싱 후 반환
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    # 파일이 없으면 빈 딕셔너리 반환
    return {}

# 사용자 정의 용어 변형 데이터를 전역 변수에 할당하고 디버그 로그 출력
CUSTOM_TERM_VARIANTS = load_custom_term_variants()
print("[DEBUG] CUSTOM_TERM_VARIANTS:", CUSTOM_TERM_VARIANTS)

# mBART 모델과 토크나이저를 로드하는 함수 (한번 로드된 후 캐시 사용)
def load_model():
    if "mbart" not in loaded:
        print("Loading mBART-50 model from local path...")
        # 로컬 경로에서 토크나이저 로드
        tokenizer = MBart50TokenizerFast.from_pretrained("./local/mbart")
        # 로컬 경로에서 모델 로드
        model = MBartForConditionalGeneration.from_pretrained("./local/mbart")
        # 로드된 토크나이저와 모델을 캐시에 저장
        loaded["mbart"] = (tokenizer, model)
    return loaded["mbart"]

# 번역 전 특정 단어를 다른 단어로 대체하는 함수
def pre_translate_replace(text):
    # 미리 정의된 치환 사전
    replacements = {
        "영진전문대학교": "Yeungjin University",
        "한국어교육센터": "Korean Education Center"
    }
    # text 내에 치환 대상 단어가 있는 경우 해당 단어를 미리 정의된 값으로 대체
    for k, v in replacements.items():
        if k in text:
            print(f"[pre_translate_replace] replacing '{k}' with '{v}'")
            text = text.replace(k, v)
    return text

# 번역 결과에서 사용자 정의 용어의 오타나 변형을 수정하는 함수
def fuzzy_replace(text, correct, variants, threshold=90):
    """
    text: 번역된 텍스트
    correct: 올바른 용어
    variants: 올바른 용어의 여러 변형(오타 등)
    threshold: 유사도 임계치 (기본 90%)
    """
    for variant in variants:
        # 먼저 변형 단어가 텍스트에 완전히 포함되어 있는지 검사
        if variant in text:
            print(f"[fuzzy_replace] exact match replacing '{variant}' with '{correct}'")
            # 정규식을 이용해 정확한 변형을 올바른 용어로 대체
            return re.sub(re.escape(variant), correct, text)
        # 그렇지 않다면 텍스트의 각 단어와 비교하여 유사도 검사
        for word in text.split():
            ratio = SequenceMatcher(None, variant.lower(), word.lower()).ratio() * 100
            if ratio >= threshold:
                print(f"[fuzzy_replace] fuzzy matched word '{word}' to '{correct}' (score: {ratio:.2f})")
                return text.replace(word, correct)
    # 대체할 변형이 없으면 원래 텍스트를 그대로 반환
    return text

# 주어진 텍스트를 target_lang (예: "en")으로 번역하는 함수
def translate_text(text, target_lang):
    print("[translate_text] original:", text)
    # 캐시에서 mBART 토크나이저와 모델을 불러옵니다.
    tokenizer, model = load_model()
    # 한국어를 source 언어로 고정 (번역 전 텍스트가 한국어임)
    src_lang = language_codes["ko"]
    # target_lang에 해당하는 토큰을 language_codes 딕셔너리에서 가져옴
    tgt_lang = language_codes.get(target_lang)
    tokenizer.src_lang = src_lang

    # 번역 전 사전 치환 실행
    text = pre_translate_replace(text)

    # 토크나이저를 이용해 텍스트를 토큰화 (패딩과 트렁케이션 적용)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # 모델을 이용해 번역 결과 토큰을 생성 (강제 초기 토큰으로 target 언어 토큰 사용)
    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang)
    )
    # 생성된 토큰을 다시 문자열로 디코딩 (특수 토큰은 제외)
    translated = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    print("[translate_text] translated before fix:", translated)

    # 사용자 정의 용어 변형에 따라 번역 결과를 수정합니다.
    try:
        for correct, variants in CUSTOM_TERM_VARIANTS.items():
            updated = fuzzy_replace(translated, correct, variants)
            if updated != translated:
                translated = updated
    except Exception as e:
        print("[translate_text] fuzzy_replace error:", e)

    print("[translate_text] translated after fix:", translated)
    return translated

# HTML 문서 내의 모든 텍스트 노드를 번역하는 함수
def translate_html(html, target):
    print("[translate_html] input html:", html)
    # BeautifulSoup을 사용해 HTML 파싱
    soup = BeautifulSoup(html, "html.parser")
    # 모든 텍스트 요소를 순회하며 번역 진행
    for tag in soup.find_all(text=True):
        # 공백만 있는 태그는 제외
        if tag.strip():
            print("[translate_html] tag:", tag)
            # 각 텍스트 노드를 번역
            translated = translate_text(tag, target)
            # 번역된 텍스트로 기존 텍스트를 대체
            tag.replace_with(translated)
    # 번역된 HTML 문자열 반환
    return str(soup)

# POST 요청을 처리하는 번역 API 엔드포인트
@app.post('/translate')
async def translate_endpoint(body: HtmlBody, response: Response):
    # 요청 바디에서 HTML 내용을 가져옴
    html = body.content
    # 현재 target 언어를 영어("en")로 설정 (필요시 다른 언어로 확장 가능)
    target = 'en'
    # HTML 내용이 없는 경우 400 에러 반환
    if not html:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"error": "Missing content"}
    try:
        # HTML 전체를 번역
        translated_html = translate_html(html, target)
        response.status_code = status.HTTP_200_OK
        return {"translated_html": translated_html}
    except Exception as e:
        # 번역 과정 중 에러 발생 시 500 에러 반환
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error": str(e)}

# 기본 GET 엔드포인트 (서버 상태 확인용)
@app.get("/")
async def root():
    return {"message": "Hello World"}
