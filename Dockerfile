# -----------------------------
# Base Image
# -----------------------------
FROM python:3.10-slim

# -----------------------------
# Set Working Directory
# -----------------------------
WORKDIR /app

# -----------------------------
# Copy Requirements First
# -----------------------------
COPY requirements.txt .

# Install Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Copy Project Files
# -----------------------------
COPY . .

# -----------------------------
# Expose Port
# -----------------------------
EXPOSE 8000

# -----------------------------
# Run FastAPI with Uvicorn
# -----------------------------
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]