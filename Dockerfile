FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pydantic \
    websockets

# Copy source code
COPY src/ /app/src/
COPY server/ /app/server/

# Expose port
EXPOSE 8000

# Run the server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
