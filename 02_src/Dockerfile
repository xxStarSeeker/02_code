FROM python:3.12-slim

WORKDIR /app

# libgomp1 = OpenMP runtime needed by faiss-cpu / torch / scikit-learn on slim images
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# PyTorch is intentionally not pinned in requirements.txt (the dev box uses a CUDA
# build). For a portable, self-contained image install the CPU build, then the rest.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so the container runs with NO first-request
# download — both the classifier's BGE component and the recommender use it.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')"

# App code + trained models + precomputed embeddings/index (see .dockerignore)
COPY . .

EXPOSE 8000 8501

# Run the API (8000) and the Streamlit demo (8501) together.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000 & streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=8501"]
