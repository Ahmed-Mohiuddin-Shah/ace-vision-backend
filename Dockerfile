FROM python:3.11.13-slim

WORKDIR /app

# Install uv (fast Python package manager)
RUN pip install uv

# Copy dependencies
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --system

# Copy code
COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
