FROM python:3.12-slim
WORKDIR /app
COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt
COPY api/curbcut api/curbcut
COPY data/pilot data/pilot
ENV PYTHONPATH=/app/api PYTHONUNBUFFERED=1
EXPOSE 8109
CMD ["uvicorn", "curbcut.main:app", "--host", "0.0.0.0", "--port", "8109"]
