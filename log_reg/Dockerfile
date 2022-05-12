FROM python:3.7
COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src /app/src
ENV PYTHONPATH=/app
WORKDIR /app/
RUN mkdir -p models

RUN python -m src.train_model

EXPOSE 8000
ENTRYPOINT ["uvicorn"]
CMD ["src.predict:app", "--host", "0.0.0.0"]


