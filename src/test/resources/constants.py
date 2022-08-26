MODELS = [{"model": 'deepset/roberta-base-squad2'}]
MODEL_ROOT = "./models_test"
CONTENTTYPE = 'application/json'
TINY_DISTILBERT_MODEL = "sshleifer/tiny-distilbert-base-cased-distilled-squad"
BERT_TINY_MODEL = "prajjwal1/bert-tiny"
MODEL_ROOT_TRAINED = "./models_test/trained/"
DATA_TEST = ["Ground spice commonly used in Indian cooking and drinks, in Middle Eastern cooking and in Scandinavian "
             "baking",
             "Alcoholic beverage produced by fermentation of grape juice produced from the chardonnay grape, "
             "with little contact with grape skins so that the wine is pale yellow in colour with little sugars "
             "remaining. Includes wine produced in Australia. Approximately 13% v/v alcohol."]

PREDICT_ENDPOINT = "/predict"
TRAIN_ENDPOINT = "/train"
DOWNLOAD_MODEL_ENDPOINT = "/download-model"
