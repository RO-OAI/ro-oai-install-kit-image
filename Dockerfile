FROM quay.io/jupyter/scipy-notebook:python-3.11

RUN mamba install --yes \
    'matplotlib<=3.7.4' \
    'seaborn' \
    'pytorch' \
    'catboost' \
    'opencv' \
    'xgboost' \
    'transformers' \
    'spacy' \
    'nltk' \
    'gensim' \
    'fasttext' \
    'lightgbm' \
    'torchvision' \
    'autoviz' \
    'datasets' \
    'evaluate' \
    'pytorch-lightning' \
    'tensorboard' \
    'torchmetrics' \
    && mamba clean --all -f -y \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"

# Add common dev tools for devcontainer compatibility
USER root
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    git \
    openssh-client \
    curl \
    wget \
    unzip \
    zip \
    sudo \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && echo "${NB_USER} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${NB_USER} \
    && chmod 0440 /etc/sudoers.d/${NB_USER}

# Pre-download NLP models for offline use
USER ${NB_UID}
ENV OFFLINE_NLP_DIR="/home/${NB_USER}/offline_nlp_assets"
ENV NLTK_DATA="${OFFLINE_NLP_DIR}/nltk_data" \
    HF_HOME="${OFFLINE_NLP_DIR}/hf_home" \
    GENSIM_DATA_DIR="${OFFLINE_NLP_DIR}/gensim_data" \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

COPY --chown=${NB_UID}:${NB_GID} prepare_nlp_offline.py /tmp/prepare_nlp_offline.py
RUN TRANSFORMERS_OFFLINE=0 HF_HUB_OFFLINE=0 /opt/conda/bin/python /tmp/prepare_nlp_offline.py \
    && rm /tmp/prepare_nlp_offline.py \
    && fix-permissions "${OFFLINE_NLP_DIR}"

# Add test notebook
USER root
COPY test_imports.ipynb /home/${NB_USER}/test_imports.ipynb
RUN fix-permissions "/home/${NB_USER}/test_imports.ipynb"
USER ${NB_UID}
