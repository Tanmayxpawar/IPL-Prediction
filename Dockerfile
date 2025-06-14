# Use a lightweight Python base image
FROM python:3.12-slim-bullseye

WORKDIR /app

COPY requirements.txt .

# Install build dependencies for compiling pandas and other C-extensions
RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    gcc \
    libffi-dev \
    libpq-dev \
    libxml2-dev \
    libxslt1-dev \
    libjpeg-dev \
    zlib1g-dev \
    curl && \
    rm -rf /var/lib/apt/lists/*


# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
