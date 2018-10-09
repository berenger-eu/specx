# Use an official Python runtime as a parent image
FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install -y wget
RUN apt-get install -y g++-7
RUN apt-get install -y cmake
RUN apt-get install -y git

