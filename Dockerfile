# syntax=docker/dockerfile:1
# start by pulling the python image
FROM python:3.9-slim-buster
ADD . /forest_watchdog
WORKDIR /forest_watchdog

# copy the requirements file into the image
COPY ./requirements.txt /app/requirements.txt

# switch working directory
# WORKDIR /app

# install the dependencies and packages in the requirements file
RUN pip install -r --no-cache-dir requirements.txt

# copy every content from the local file to the image
COPY . /app

# configure the container to run in an executed manner
# ENTRYPOINT [ "python" ]

CMD ["python","./src/uvicorn main:app --reload" ]
