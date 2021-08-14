# syntax=docker/dockerfile:1.0.0-experimental
FROM amancevice/pandas:0.23.4-python3-slim

WORKDIR /Main

COPY requirements.txt ./

RUN mkdir divinity
COPY divinity/ /Main/divinity
#
RUN mkdir tests
COPY tests/ /Main/tests/
#
RUN pip install --no-cache-dir -r requirements.txt

#add divinity library to python path

#CMD ["python /Main/disease_spread/main.py"]
CMD ["pytest"]

# docker build -f Docker/Dockerfile -t ds207/divinity .
# docker run -it ds207/divinity bash
# docker push ds207/divinity

#delete all containers and images
#docker rm -vf $(docker ps -a -q)
#docker rmi -f $(docker images -a -q)
