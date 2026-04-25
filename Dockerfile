FROM python:3.11-slim

WORKDIR /app

# Install both serving and training dependencies.
COPY requirements.txt requirements-train.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-train.txt

# Copy all project files and set ownership to UID 1000 in one atomic step.
COPY --chown=1000:1000 . .

# Switch to the non-root user required by Hugging Face.
USER 1000

EXPOSE 7860

# For the training Space, boot API + GRPO runner.
CMD ["sh", "/app/run_training_space.sh"]