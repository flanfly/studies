from asia-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-3.py310:latest

workdir /app

copy requirements.txt .

run pip install -r requirements.txt --no-cache-dir

copy . .

entrypoint ["python", "vertex-ai-main.py"]
