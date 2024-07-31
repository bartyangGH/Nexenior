# Use an official Python runtime as a parent image
FROM python:3.11-slim-buster

# https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
# These commands install the cv2 dependencies that are normally present on the local machine, but might be missing in your Docker container causing the issue.
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Set the working directory in the container to /app
WORKDIR /app

# Install any needed packages specified in requirements.txt
ADD ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Add the current directory contents into the container at /app
ADD . /app

# Make port 80 available to the world outside this container
EXPOSE 80

# Run YOLO_v8.py when the container launches
CMD ["python", "YOLO_v8.py"]