FROM python:3.11-slim


WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
COPY requirements-dev.txt .

RUN pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.12.1 \
    torchvision==0.27.1

RUN pip install -r requirements.txt --no-deps
RUN pip install -r requirements-dev.txt

COPY src ./src
COPY data ./data
COPY tests ./tests
ENV PYTHONPATH=/app

COPY start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]