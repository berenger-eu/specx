FROM ubuntu:22.04

RUN apt-get update \
    && apt-get install -y apt-transport-https \
    && apt-get install -y wget \
    && apt-get install -y g++ \
    && apt-get install -y git \
    && apt-get install -y make \
    && apt-get install -y cmake \
    && apt-get install -y clang \
    && apt-get install -y lcov

# To rebuild and update the repo:
# - Make sure there are not images: sudo docker images
# - Remove all images: sudo docker rmi --force c7885369373a
# - Build the new image: sudo docker build -t specx .
# - Tag it: sudo docker tag 257399324d18 berenger/specxci:latest
# - Push it: sudo docker push berenger/specxci
# - Test it: sudo docker exec -it berenger/specxci:latest /bin/bash
#            or sudo docker run -it berenger/specxci:latest /bin/bash
