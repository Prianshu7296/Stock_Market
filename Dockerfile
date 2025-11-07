# Use the latest Python image as a base
FROM python:3.14

# Set working directory inside the container
WORKDIR /app

# Copy all contents of your project into the container
COPY . /app

# Upgrade pip and install all dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Default command to run your pipeline script
CMD ["python", "main.py"]
