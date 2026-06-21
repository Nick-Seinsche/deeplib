# Use minimal Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy only necessary files
COPY deploy.py .

# Install required packages
RUN pip install --no-cache-dir flask pandas scikit-learn

# Expose the Flask port
EXPOSE 5000

# Run the Flask app
CMD ["python", "deploy.py"]