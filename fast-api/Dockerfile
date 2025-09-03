# Use official Python image
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app files (including model.pkl and main.py)
COPY . .

# Expose port for FastAPI
EXPOSE 5000

# Command to run the FastAPI app with uvicorn
CMD ["uvicorn", "main:app_fastapi.py", "--host", "0.0.0.0", "--port", "8000"]

