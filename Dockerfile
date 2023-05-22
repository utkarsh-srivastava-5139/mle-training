FROM continuumio/miniconda3
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

LABEL maintainer="Utkarsh Srivastava"

COPY . .

RUN apt-get update \
    && apt-get install -y wget \
    && rm -rf /var/lib/apt/lists/*

# RUN set -xe \
#     && apt-get update \
#     && apt-get install -y python3-pip

# RUN pip install --upgrade pip

# RUN wget \
#     https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#     && mkdir /root/.conda \
#     && bash Miniconda3-latest-Linux-x86_64.sh -b \
#     && rm -f Miniconda3-latest-Linux-x86_64.sh 

# RUN conda env create -f deploy/conda/env.yml

# SHELL ["conda", "run", "-n", "mle-dev", "/bin/bash", "-c"]

RUN pip install housinglib-0.1.0-py3-none-any.whl 
    # && pip install -r requirements.txt

WORKDIR /scripts

ENTRYPOINT ["/bin/bash"]
