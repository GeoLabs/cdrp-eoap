FROM mambaorg/micromamba:jammy

COPY ./myenv-def.yml conda-environment.yml

ARG CONDA_ENV=myenv
RUN micromamba create -n ${CONDA_ENV} -y -f conda-environment.yml
COPY ./app.py /opt/conda/envs/${CONDA_ENV}/site-packages/app.py
ENV PATH /opt/conda/envs/${CONDA_ENV}/bin:$PATH
ENV CONDA_DEFAULT_ENV ${CONDA_ENV}
