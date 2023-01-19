# syntax=docker/dockerfile:1
# start by pulling the python image
FROM python:3.9-slim-buster
ADD . /forest_watchdog
WORKDIR /forest_watchdog

# copy the requirements file into the image
COPY ./requirements.txt /app/requirements.txt

# switch working directory
# WORKDIR /app

RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

# install the dependencies and packages in the requirements file
RUN pip install --no-cache-dir -r requirements.txt 
# copy every content from the local file to the image
COPY . /app

# configure the container to run in an executed manner
# ENTRYPOINT [ "python" ]

# CMD ["python","./src/main.py" ]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3137"]