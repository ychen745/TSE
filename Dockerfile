# Python 3.13, app factory at app:create_app()
FROM python:3.13-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
  && rm -rf /var/lib/apt/lists/*
RUN pip install -U pip wheel setuptools
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps -r requirements.txt -w /wheels

FROM python:3.13-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8080 \
    HOME=/home/app MPLCONFIGDIR=/home/app/.config/matplotlib MPLBACKEND=Agg
RUN addgroup --system app && adduser --system --home /home/app --ingroup app app \
 && mkdir -p /home/app/.config/matplotlib && chown -R app:app /home/app
COPY --from=builder /wheels /wheels
RUN pip install --no-cache /wheels/*
COPY . .
USER app
EXPOSE 8080
ENV APP_MODULE="app:create_app()"
CMD ["sh","-c","gunicorn -w 2 -k gthread --threads 4 -b 0.0.0.0:${PORT} ${APP_MODULE} --access-logfile - --error-logfile -"]
