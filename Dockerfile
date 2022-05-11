FROM python:3.7
COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY titanic_pipeline /app/titanic_pipeline
COPY models /app/models

ENV PYTHONPATH=/app
WORKDIR /app/

EXPOSE 8000

ENTRYPOINT ["uvicorn"]
CMD ["titanic_pipeline.predict:app", "--host", "0.0.0.0", "--reload"]