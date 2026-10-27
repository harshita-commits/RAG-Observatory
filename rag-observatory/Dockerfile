FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements-live.txt requirements-documents.txt ./
RUN pip install -r requirements-live.txt -r requirements-documents.txt
COPY observatory ./observatory
COPY examples ./examples
COPY tests ./tests
RUN useradd --uid 10001 --create-home observatory && mkdir -p /app/artifacts && chown observatory:observatory /app/artifacts
USER observatory
EXPOSE 8000
CMD ["python", "-m", "observatory", "serve", "--port", "8000"]
