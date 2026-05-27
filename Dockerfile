FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /opt/tsugi-mend

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
COPY docs ./docs
COPY examples ./examples

RUN python -m pip install --upgrade pip && python -m pip install .

CMD ["python", "-c", "import tsugi_mend; print(tsugi_mend.__version__)"]
