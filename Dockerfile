from jupyter/base-notebook

USER root
RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y gcc
