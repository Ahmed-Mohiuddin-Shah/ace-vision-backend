FROM python:3.11.13-slim

# Install system dependencies for OpenCV and other Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /

# Install uv (fast Python package manager)
RUN pip install uv

# Copy dependencies
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen

# Copy code
COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
