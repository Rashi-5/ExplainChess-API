# ExplainChess API — Hugging Face Docker Space (listens on 7860, non-root).
# The 621 MB model is committed into the Space repo at models/ (HF handles large
# files natively). External Postgres + Stripe come from Space secrets.
FROM python:3.11-slim

# Stockfish pinned to the official 18 release so prod evals match local (Homebrew
# also ships SF 18). Debian apt's stockfish is older and its different NNUE net
# gives different evals at the same depth, skewing the feature vector. If the
# Space CPU lacks AVX2 (engine crashes at startup), switch to
# stockfish-ubuntu-x86-64-sse41-popcnt.
ARG SF_ASSET=stockfish-ubuntu-x86-64-avx2
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && curl -fsSL -o /tmp/sf.tar \
        "https://github.com/official-stockfish/Stockfish/releases/download/sf_18/${SF_ASSET}.tar" \
    && tar -xf /tmp/sf.tar -C /tmp \
    && mv "/tmp/stockfish/${SF_ASSET}" /usr/local/bin/stockfish \
    && chmod a+x /usr/local/bin/stockfish \
    && rm -rf /tmp/sf.tar /tmp/stockfish \
    && apt-get purge -y curl && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*
ENV STOCKFISH_PATH=/usr/local/bin/stockfish

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY engine/ ./engine/
COPY models/ ./models/
COPY config.yaml ./config.yaml

# HF Spaces run the container as an arbitrary non-root UID.
RUN chmod -R a+rX /app

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
