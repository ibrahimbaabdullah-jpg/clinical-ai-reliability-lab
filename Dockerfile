# Runtime defaults to CPU tabular models. Add --build-arg WITH_DEEP=1 for CPU GRU.
FROM python:3.12-slim
ARG WITH_DEEP=0
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md requirements-tested.txt requirements-linux-py312.txt ./
COPY clinical_lab ./clinical_lab
COPY configs ./configs
COPY docs ./docs
# Constraints reproduce the installed dependency versions used for this Linux/Python release.
# The default install does not pull PyTorch or MLflow.
RUN python -m pip install --no-cache-dir --constraint requirements-linux-py312.txt . \
    && if [ "$WITH_DEEP" = "1" ]; then \
         python -m pip install --no-cache-dir --constraint requirements-linux-py312.txt torch \
           --index-url https://download.pytorch.org/whl/cpu; \
       elif [ "$WITH_DEEP" != "0" ]; then \
         echo 'WITH_DEEP must be 0 or 1' >&2; exit 1; \
       fi \
    && python -m pip check
RUN useradd --create-home --uid 10001 lab \
    && mkdir -p /app/artifacts \
    && chown lab:lab /app/artifacts
USER lab
EXPOSE 8000
# /api/status proves model initialization succeeds, including a mounted trusted run.
HEALTHCHECK --interval=30s --timeout=30s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/status', timeout=25).read()"
# Publish only to localhost on the host: -p 127.0.0.1:8000:8000.
# A trusted artifact mount is opt-in via CLINICAL_LAB_RUN=/app/run.
CMD ["uvicorn", "clinical_lab.api:app", "--host", "0.0.0.0", "--port", "8000"]
