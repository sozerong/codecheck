# Python 3.10-slim 기반 이미지
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 설치를 위한 파일 복사
COPY ./requirements.txt /tmp/requirements.txt

# Python 의존성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Django 프로젝트 관련 파일 복사
COPY ./manage.py /app/
COPY ./llm_project /app/llm_project/

# 컨테이너 실행 시 기본 명령어
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
