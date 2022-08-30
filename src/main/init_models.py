"""
NOTE: This file is only meant to run locally to download models!
"""
from src.main.download import download_models

models = {
    "questionAndAnswer": ['deepset/roberta-base-squad2',
                          'mrm8488/distilbert-multi-finetuned-for-xqua-on-tydiqa',
                          'distilbert-base-cased-distilled-squad']
}

path = "../models"

download_models(models, path)
