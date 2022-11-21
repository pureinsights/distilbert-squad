MODELS_QA = {"questionAndAnswer": ['deepset/roberta-base-squad2']}
MODELS_ST = {"sentenceTransformer": ['deepset/roberta-base-squad2']}
MODEL_ROOT = "./models_test"
MODEL_ROOT_QA = "./models_test/questionAndAnswer"
MODEL_ROOT_ST = "./models_test/sentenceTransformer"

CONTENTTYPE = 'application/json'

MODEL_ROOT_TRAINED = "/models_test/trained/"
DATA_TEST = ["Ground spice commonly used in Indian cooking and drinks, in Middle Eastern cooking and in Scandinavian "
             "baking",
             "Alcoholic beverage produced by fermentation of grape juice produced from the chardonnay grape, "
             "with little contact with grape skins so that the wine is pale yellow in colour with little sugars "
             "remaining. Includes wine produced in Australia. Approximately 13% v/v alcohol."]
DATA_TEST_SINGLE_TEXT = [
    'The FA Cup is open to any eligible club down to Level 10 of the English football league system – 20 '
    'professional clubs in the Premier League (level 1),72 professional clubs in the English Football '
    'League (levels 2 to 4), and several hundred non-League teams in steps 1 to 6 of the National League '
    'System (levels 5 to 10). A record 763 clubs competed in 2011–12. The tournament consists of 12 '
    'randomly drawn rounds followed by the semi-finals and the final. Entrants are not seeded, '
    'although a system of byes based on league level ensures higher ranked teams enter in later rounds.  '
    'The minimum number of games needed to win, depending on which round a team enters the competition, '
    'ranges from six to fourteen.', 'The first six rounds are the Qualifying Competition, from which 32 '
                                    'teams progress to the first round of the Competition Proper, '
                                    'meeting the first of the 48 professional teams from Leagues One and '
                                    'Two. The last entrants are the Premier League and Championship '
                                    'clubs, into the draw for the Third Round Proper.[2] In the modern '
                                    'era, only one non-League team has ever reached the quarter-finals, '
                                    'and teams below Level 2 have never reached the final.[note 1] As a '
                                    'result, significant focus is given to the smaller teams who '
                                    'progress furthest, especially if they achieve an unlikely '
                                    '"giant-killing" victory.']

PREDICT_ENDPOINT = "/predict"
ENCODE_ENDPOINT = "/encode"
TRAIN_ENDPOINT = "/train"
DOWNLOAD_MODEL_ENDPOINT = "/download-model"

TINY_DISTILBERT_MODEL = "sshleifer/tiny-distilbert-base-cased-distilled-squad"
BERT_TINY_MODEL = "prajjwal1/bert-tiny"
BERT_UNCASED_MODEL = "bert-base-uncased"

PATHS_TO_DOWNLOAD = {
    "sentencetransformer": MODEL_ROOT_ST,
    "questionandanswer": MODEL_ROOT_QA,
    "default": MODEL_ROOT_QA
}
