# Use an official Python runtime as a parent image
FROM python:3.6

# Set the working directory to /app
WORKDIR /home/bruce/workspace/mnist

# Copy the current directory contents into the container at /app
COPY . /home/bruce/workspace/mnist

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py","--models","fc","--methods","train"]