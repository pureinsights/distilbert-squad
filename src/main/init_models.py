"""
NOTE: This file is only meant to run locally to download models!
"""
from download import download_models

models = {
    "sentenceTransformer": ['sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2']
}

paths = {
    "default": "../models"
}




download_models(paths,models)
