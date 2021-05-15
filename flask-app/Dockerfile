FROM python:3.6-slim

WORKDIR /deploy/

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5100

ENTRYPOINT ["python", "cancer-detection.py"]
