from jupyter/base-notebook

USER root
RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y gcc
RUN apt-get install -y libfreetype6-dev
RUN apt-get install -y libpng-dev
RUN apt-get install -y pkg-config
RUN apt-get install -y libpq-dev 
RUN conda install -c conda-forge -y pygpu

COPY ./requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt
