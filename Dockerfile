# docker build -t fast_demo_img .
FROM python:3.11-slim

WORKDIR /app

COPY models/observation_bounds.joblib models/final_model.joblib models/
COPY src/__init__.py src/main.py src/

COPY requirements.txt app.py ./


RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["fastapi", "run", "app.py"]

