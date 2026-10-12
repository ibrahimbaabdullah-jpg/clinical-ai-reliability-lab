FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY clinical_lab ./clinical_lab
COPY configs ./configs
COPY docs ./docs
RUN pip install --no-cache-dir .
ARG WITH_DEEP=0
RUN if [ "$WITH_DEEP" = "1" ]; then pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; fi
RUN useradd --create-home lab && mkdir -p /app/artifacts && chown -R lab:lab /app
USER lab
EXPOSE 8000
CMD ["uvicorn", "clinical_lab.api:app", "--host", "0.0.0.0", "--port", "8000"]
