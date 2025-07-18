FROM python:3.12.9-bullseye

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

# Comando para ejecutar FastAPI
CMD ["fastapi", "run", "app/main.py"]