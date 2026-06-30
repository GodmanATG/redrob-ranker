FROM python:3.11-slim

# Set environment variables to force offline execution during rank.py
ENV HUGGINGFACE_HUB_CACHE=/hf_cache
ENV HF_HOME=/hf_cache
ENV TRANSFORMERS_OFFLINE=1

WORKDIR /app

# 1. Install CPU-only PyTorch first (drastically reduces image size from ~3GB to ~500MB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Pre-download the AI model weights into the Docker image 
# This guarantees 0 network calls during the actual execution (hackathon requirement)
ENV TRANSFORMERS_OFFLINE=0
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', cache_folder='/hf_cache')"
ENV TRANSFORMERS_OFFLINE=1

# 4. Copy source code
COPY . .

# 5. Define entrypoint
ENTRYPOINT ["python", "-X", "utf8", "rank.py"]
CMD ["--candidates", "/data/candidates.jsonl", "--out", "/data/submission.csv"]
