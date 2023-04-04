# az acr build -r crjupyterhub -f Dockerfile -t vvcb/fftsentiment:dev -t vvcb/fftsentiment:0.1.0 .
# docker build -f Dockerfile -t vvcb/fftsentiment:dev
FROM python:3.8-slim-buster

# Set Environment variables
# Path to models
ENV DIR_MODELS="/home/appuser/models"
# Set path to stop pip complaining during installation
# This can also be done after the pip install step
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Create a non-root user for running the container and API
RUN groupadd -g 999 appuser \
    && useradd --create-home -r -u 999 -g appuser appuser

# Switch to non-root user
USER appuser
WORKDIR /home/appuser

# Install requirements
COPY --chown=appuser:appuser requirements.txt requirements.txt
RUN pip3 install --user --no-cache-dir -r requirements.txt

# Copy over models and code
COPY --chown=appuser:appuser ./models ./models
COPY --chown=appuser:appuser ./fft_api ./fft_api

# Expose port for API and start API
EXPOSE 5000
CMD cd fft_api && uvicorn main:app --host 0.0.0.0 --port 5000
