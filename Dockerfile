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

# Add test notebook
USER root
COPY test_imports.ipynb /home/${NB_USER}/test_imports.ipynb
RUN fix-permissions "/home/${NB_USER}/test_imports.ipynb"
USER ${NB_UID}
