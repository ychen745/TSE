# ---------- builder ----------
FROM python:3.13-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
  && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip wheel setuptools
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps -r requirements.txt -w /wheels

# ---------- runtime ----------
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080
WORKDIR /app

# Create user with a real home
RUN addgroup --system app && adduser --system --home /home/app --ingroup app app
# Matplotlib config (avoid /nonexistent warnings)
RUN mkdir -p /home/app/.config/matplotlib && chown -R app:app /home/app
ENV HOME=/home/app \
    MPLCONFIGDIR=/home/app/.config/matplotlib \
    MPLBACKEND=Agg

COPY --from=builder /wheels /wheels
RUN pip install --no-cache /wheels/*
COPY . .
USER app
EXPOSE 8080

# Make entrypoint configurable; set a sensible default
CMD ["gunicorn", "-w", "2", "-k", "gthread", "--threads", "4", "-b", "0.0.0.0:8080", "app:create_app()"]