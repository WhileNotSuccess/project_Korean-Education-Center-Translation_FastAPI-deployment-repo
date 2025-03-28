# 1. Python 공식 이미지 기반
FROM python:3.10-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 의존성 관련 파일 복사
COPY requirements.txt .

# 4. 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 5. 코드 복사
COPY . .

# 6. 포트 노출 (FastAPI 기본 포트)
EXPOSE 5000

# 7. Gunicorn을 사용한 FastAPI 실행
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120"]
