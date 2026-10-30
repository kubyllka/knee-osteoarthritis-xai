FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY notebooks/kaggle_research_runner.ipynb ./notebooks/kaggle_research_runner.ipynb
COPY research_toolkit/README.md ./research_toolkit/README.md
COPY .streamlit/config.toml ./.streamlit/config.toml
COPY streamlit_app.py README.md ./
RUN mkdir -p models inputs results/research

EXPOSE 8000 8501

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
