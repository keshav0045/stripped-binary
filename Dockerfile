# Serving image (kNN model by default; mount a LoRA adapter and install
# requirements-llm.txt to serve the fine-tuned model instead).
#   docker build -t fnname . && docker run -p 8000:8000 fnname
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY fnname fnname
COPY serve serve
COPY models/knn.pkl models/knn.pkl
ENV FNNAME_MODEL=models/knn.pkl
EXPOSE 8000
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
