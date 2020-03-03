# Use an official Python runtime as a parent image
FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install -y apt-transport-https
RUN apt-get install -y wget
RUN apt-get install -y g++
RUN apt-get install -y git
RUN apt-get install make
RUN wget https://github.com/Kitware/CMake/releases/download/v3.15.6/cmake-3.15.6-Linux-x86_64.sh && chmod +x cmake-3.15.6-Linux-x86_64.sh && ./cmake-3.15.6-Linux-x86_64.sh --skip-license --prefix=/usr/

# To rebuild and update the repo:
# - Make sure there are not images: sudo docker images
# - Remove all images: sudo docker rmi --force c7885369373a
# - Build the new image: sudo docker build -t spetabaru .
# - Tag it: sudo docker tag 257399324d18 berenger/spetabaruci:latest
# - Push it: sudo docker push berenger/spetabaruci
# - Test it: sudo docker exec -it berenger/spetabaruci:latest /bin/bash
#            or sudo docker run -it berenger/spetabaruci:latest /bin/bash
