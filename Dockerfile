# 1. Base Python image
FROM python:3.10

# 2. Set working directory inside container
WORKDIR /app

# 3. Copy all project files into container
COPY . /app

# 4. Install required libraries
RUN pip install --no-cache-dir -r requirements.txt

# 5. Expose Flask port
EXPOSE 5000

# 6. Run Flask app
CMD ["python", "app.py"]