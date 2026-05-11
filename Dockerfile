FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose ports (FastAPI: 8000, Streamlit: 8501)
EXPOSE 8000 8501

# Create startup script
RUN echo '#!/bin/bash\n\
python backend/main.py &\n\
sleep 5\n\
streamlit run frontend/dashboard.py --server.port=8501 --server.address=0.0.0.0\n\
' > start.sh && chmod +x start.sh

CMD ["./start.sh"]
