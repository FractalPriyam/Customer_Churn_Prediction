FROM python:3.10.5-slim

# 2. Set working directory
WORKDIR /app

# 3. Install System Dependencies
# 'build-essential' and 'libgomp1' are critical for XGBoost/LightGBM
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy Requirements & Install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Source Code & Config
COPY src/ ./src/
COPY config.yaml .
COPY data/ ./data/
COPY exported_model/ ./exported_model/

# 6. Copy the OFFLINE MODEL (The folder created by export_model.py)
# COPY docker_model/ ./docker_model/

ENV PYTHONPATH="/usr/local/bin/python:/app/src"

# 7. Expose Port
EXPOSE 8100

# 8. Run the API
CMD ["uvicorn", "app:app", "--app-dir","src", "--host", "0.0.0.0", "--port", "8100"]