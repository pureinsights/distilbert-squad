import shutil
import unittest
from unittest import mock

import numpy as np
from flask import Flask
from sys import path
import os
import json
import spacy

path.insert(0, os.getcwd() + "/../")
import src.main.huggingface as huggingface
import src.main.model as model

from transformers import AutoModelForMaskedLM, AutoTokenizer

app = Flask(__name__)
app.register_blueprint(huggingface.huggingface_api)

models = [{"model": 'deepset/roberta-base-squad2'}]
model_root = "./models_test"

model.download_models(models, model_root)


class TestEndpoint(unittest.TestCase):
    tester = None

    def __init__(self, *args, **kwargs):
        super(TestEndpoint, self).__init__(*args, **kwargs)
        global tester
        tester = app.test_client()

    def setUp(self):
        huggingface.model = model.Model(model_root)

    @mock.patch('huggingface.input')
    def test_models(self, mock_input):
        response = tester.get(
            '/models',
            content_type='application/json'
        )

        data = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data, '["deepset/roberta-base-squad2"]')

    @mock.patch('huggingface.input')
    def test_predict(self, mock_input):
        body = {
            "model": "sshleifer/tiny-distilbert-base-cased-distilled-squad",
            "question": "How many games are required to win the FA Cup?",
            "chunks": [
                {
                    "text": "The FA Cup is open to any eligible club down to Level 10 of the English football league system – 20 professional clubs in the Premier League (level 1),72 professional clubs in the English Football League (levels 2 to 4), and several hundred non-League teams in steps 1 to 6 of the National League System (levels 5 to 10). A record 763 clubs competed in 2011–12. The tournament consists of 12 randomly drawn rounds followed by the semi-finals and the final. Entrants are not seeded, although a system of byes based on league level ensures higher ranked teams enter in later rounds.  The minimum number of games needed to win, depending on which round a team enters the competition, ranges from six to fourteen.",
                    "id": "1"
                },
                {
                    "text": "The first six rounds are the Qualifying Competition, from which 32 teams progress to the first round of the Competition Proper, meeting the first of the 48 professional teams from Leagues One and Two. The last entrants are the Premier League and Championship clubs, into the draw for the Third Round Proper.[2] In the modern era, only one non-League team has ever reached the quarter-finals, and teams below Level 2 have never reached the final.[note 1] As a result, significant focus is given to the smaller teams who progress furthest, especially if they achieve an unlikely \"giant-killing\" victory.",
                    "id": "2"
                }
            ]
        }

        answer = '[{"score": 0.6043325066566467, "start": 693, "end": 708, "answer": "six to fourteen", ' \
                 '"id": "1", "highlight": "The FA Cup is open to any eligible club down to Level 10 of ' \
                 'the English football league system \\u2013 20 professional clubs in the Premier League ' \
                 '(level 1),72 professional clubs in the English Football League (levels 2 to 4), ' \
                 'and several hundred non-League teams in steps 1 to 6 of the National League System (' \
                 'levels 5 to 10). A record 763 clubs competed in 2011\\u201312. The tournament consists ' \
                 'of 12 randomly drawn rounds followed by the semi-finals and the final. Entrants are ' \
                 'not seeded, although a system of byes based on league level ensures higher ranked ' \
                 'teams enter in later rounds.  The minimum number of games needed to win, depending on ' \
                 'which round a team enters the competition, ranges from <span class=\\\"highlight\\\">six ' \
                 'to fourteen</span>."}, {"score": 0.0016566978301852942, "start": 64, "end": 66, ' \
                 '"answer": "32", "id": "2", "highlight": "The first six rounds are the Qualifying ' \
                 'Competition, from which <span class=\\\"highlight\\\">32</span> teams progress to the ' \
                 'first round of the Competition Proper, meeting the first of the 48 professional teams ' \
                 'from Leagues One and Two. The last entrants are the Premier League and Championship ' \
                 'clubs, into the draw for the Third Round Proper.[2] In the modern era, ' \
                 'only one non-League team has ever reached the quarter-finals, and teams below Level 2 ' \
                 'have never reached the final.[note 1] As a result, significant focus is given to the ' \
                 'smaller teams who progress furthest, especially if they achieve an unlikely ' \
                 '\\\"giant-killing\\\" victory."}]'

        response = tester.post(
            '/predict',
            content_type='application/json',
            json=body
        )

        data = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data, answer)

    def test_predict_error_message(self):
        body_no_question = {
            "model": "sshleifer/tiny-distilbert-base-cased-distilled-squad",
            "chunks": [
                {
                    "text": "The FA Cup is open to any eligible club down to Level 10 of the English football league system – 20 professional clubs in the Premier League (level 1),72 professional clubs in the English Football League (levels 2 to 4), and several hundred non-League teams in steps 1 to 6 of the National League System (levels 5 to 10). A record 763 clubs competed in 2011–12. The tournament consists of 12 randomly drawn rounds followed by the semi-finals and the final. Entrants are not seeded, although a system of byes based on league level ensures higher ranked teams enter in later rounds.  The minimum number of games needed to win, depending on which round a team enters the competition, ranges from six to fourteen.",
                    "id": "1"
                },
                {
                    "text": "The first six rounds are the Qualifying Competition, from which 32 teams progress to the first round of the Competition Proper, meeting the first of the 48 professional teams from Leagues One and Two. The last entrants are the Premier League and Championship clubs, into the draw for the Third Round Proper.[2] In the modern era, only one non-League team has ever reached the quarter-finals, and teams below Level 2 have never reached the final.[note 1] As a result, significant focus is given to the smaller teams who progress furthest, especially if they achieve an unlikely \"giant-killing\" victory.",
                    "id": "2"
                }
            ]
        }

        body_no_chunks = {
            "model": "sshleifer/tiny-distilbert-base-cased-distilled-squad",
            "question": "How many games are required to win the FA Cup?"
        }

        response = tester.post(
            '/predict',
            content_type='application/json',
            json=None
        )

        self.assertEqual(response.status_code, 400)

        response = tester.post(
            '/predict',
            content_type='application/json',
            json=body_no_question
        )

        data = json.loads(response.get_data(as_text=True))
        self.assertEqual(data["message"], 'The prediction needs a question')
        self.assertEqual(response.status_code, 400)

        response = tester.post(
            '/predict',
            content_type='application/json',
            json=body_no_chunks
        )

        data = json.loads(response.get_data(as_text=True))
        self.assertEqual(data["message"], 'The prediction needs at least one chunk')
        self.assertEqual(response.status_code, 400)

    def test_status_with_no_train(self):
        response = tester.get(
            '/status',
            content_type='application/json'
        )

        data = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data, '{"training_status": false}')

    def test_train(self):
        body = {
            "output_path": "./models_test/trained/",
            "model": "bert-base-uncased",
            "data": [
                "Ground spice commonly used in Indian cooking and drinks, in Middle Eastern cooking and in Scandinavian baking",
                "Alcoholic beverage produced by fermentation of grape juice produced from the chardonnay grape, with little contact with grape skins so that the wine is pale yellow in colour with little sugars remaining. Includes wine produced in Australia. Approximately 13% v/v alcohol."
            ],
            "batch_training": 2
        }

        response = tester.post(
            '/train',
            content_type='application/json',
            json=body
        )

        data = json.loads(response.get_data(as_text=True))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["message"], "Training started")

    def test_train_error_message(self):
        body_no_output_path = {
            "model": "bert-base-uncased",
            "data": [
                "Ground spice commonly used in Indian cooking and drinks, in Middle Eastern cooking and in Scandinavian baking",
                "Alcoholic beverage produced by fermentation of grape juice produced from the chardonnay grape, with little contact with grape skins so that the wine is pale yellow in colour with little sugars remaining. Includes wine produced in Australia. Approximately 13% v/v alcohol."
            ],
            "batch_training": 2
        }

        body_no_model = {
            "output_path": "./models_test/trained/",
            "data": [
                "Ground spice commonly used in Indian cooking and drinks, in Middle Eastern cooking and in Scandinavian baking",
                "Alcoholic beverage produced by fermentation of grape juice produced from the chardonnay grape, with little contact with grape skins so that the wine is pale yellow in colour with little sugars remaining. Includes wine produced in Australia. Approximately 13% v/v alcohol."
            ],
            "batch_training": 2
        }

        body_no_data = {
            "output_path": "./models_test/trained/",
            "model": "bert-base-uncased",
            "batch_training": 2
        }

        body_data_empty = {
            "output_path": "./models_test/trained/",
            "model": "bert-base-uncased",
            "data": None,
            "batch_training": 2
        }

        response = tester.post(
            '/train',
            content_type='application/json',
            json=None
        )

        self.assertEqual(response.status_code, 400)

        response = tester.post(
            '/train',
            content_type='application/json',
            json=body_no_output_path
        )

        data = json.loads(response.get_data(as_text=True))
        self.assertEqual(data["message"], 'Missing parameter output_path')
        self.assertEqual(response.status_code, 400)

        response = tester.post(
            '/train',
            content_type='application/json',
            json=body_no_model
        )

        data = json.loads(response.get_data(as_text=True))
        self.assertEqual(data["message"], 'Missing parameter model')
        self.assertEqual(response.status_code, 400)

        response = tester.post(
            '/train',
            content_type='application/json',
            json=body_no_data
        )

        data = json.loads(response.get_data(as_text=True))
        self.assertEqual(data["message"], 'Missing parameter data')
        self.assertEqual(response.status_code, 400)

        response = tester.post(
            '/train',
            content_type='application/json',
            json=body_data_empty
        )

        data = json.loads(response.get_data(as_text=True))
        self.assertEqual(data["message"], 'Missing information in parameter data')
        self.assertEqual(response.status_code, 400)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(model_root)


class TestEndpointFunctions(unittest.TestCase):
    tester = None

    def __init__(self, *args, **kwargs):
        super(TestEndpointFunctions, self).__init__(*args, **kwargs)
        global tester
        tester = app.test_client()

    def setUp(self):
        huggingface.nlp = spacy.load("en_core_web_sm",
                                     exclude=['morphologizer', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])

    def test_highlight(self):
        text = "The FA Cup is open to any eligible club down to Level 10 of the English football league system – 20 " \
               "professional clubs in the Premier League (level 1),72 professional clubs in the English Football " \
               "League (levels 2 to 4), and several hundred non-League teams in steps 1 to 6 of the National League " \
               "System (levels 5 to 10). A record 763 clubs competed in 2011–12. The tournament consists of 12 " \
               "randomly drawn rounds followed by the semi-finals and the final. Entrants are not seeded, although a " \
               "system of byes based on league level ensures higher ranked teams enter in later rounds.  The minimum " \
               "number of games needed to win, depending on which round a team enters the competition, ranges from " \
               "six to fourteen. "
        answer = {'score': 4.5736833271803334e-05, 'start': 0, 'end': 3, 'answer': 'The', 'id': '1'}
        style = "highlight"

        response = '<span class="highlight">The</span> FA Cup is open to any ' \
                   'eligible club down to Level 10 of the English football league system – 20 professional clubs in ' \
                   'the Premier League (level 1),72 professional clubs in the English Football League (levels 2 to ' \
                   '4), and several hundred non-League teams in steps 1 to 6 of the National League System (levels 5 ' \
                   'to 10). A record 763 clubs competed in 2011–12. The tournament consists of 12 randomly drawn ' \
                   'rounds followed by the semi-finals and the final. Entrants are not seeded, although a system of ' \
                   'byes based on league level ensures higher ranked teams enter in later rounds.  The minimum number ' \
                   'of games needed to win, depending on which round a team enters the competition, ranges from six ' \
                   'to fourteen. '

        result = huggingface.highlight(text, answer, style)

        self.assertEqual(result, response)

    def test_start_train(self):
        model_name = "bert-base-uncased"
        output_path = "./models_test/trained/"
        batch_size = 2
        data = [
            "Ground spice commonly used in Indian cooking and drinks, in Middle Eastern cooking and in Scandinavian baking",
            "Alcoholic beverage produced by fermentation of grape juice produced from the chardonnay grape, with little contact with grape skins so that the wine is pale yellow in colour with little sugars remaining. Includes wine produced in Australia. Approximately 13% v/v alcohol."
        ]

        answer = {"message": "MLM Training finished, model saved at: './models_test/trained/'",
                  "added_tokens": ["Alcoholic", "Approximately", "Australia", "Eastern", "Ground", "Includes", "Indian",
                                   "Middle", "Scandinavian", "chardonnay", "fermentation", "sugars"]}

        response = huggingface.start_train(model_name, output_path, batch_size, data)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        del data["timestamp"]
        self.assertEqual(data, answer)
        self.assertTrue(os.path.exists("./models_test/trained") and os.listdir("./models_test/trained"))

        shutil.rmtree("./models_test/trained")

    def test_train_mlm(self):
        docs = [
            "Ground spice commonly used in Indian cooking and drinks, in Middle Eastern cooking and in Scandinavian baking",
            "Alcoholic beverage produced by fermentation of grape juice produced from the chardonnay grape, with little contact with grape skins so that the wine is pale yellow in colour with little sugars remaining. Includes wine produced in Australia. Approximately 13% v/v alcohol."
        ]
        model_name = "bert-base-uncased"
        output_path = "./models_test/trained/"

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        tokenizer.add_tokens(
            ['Alcoholic', 'Approximately', 'Australia', 'Eastern', 'Ground', 'Includes', 'Indian', 'Middle',
             'Scandinavian', 'chardonnay', 'fermentation', 'sugars'])

        loaded_model = AutoModelForMaskedLM.from_pretrained(model_name)
        loaded_model = loaded_model.to("cpu")
        loaded_model.resize_token_embeddings(len(tokenizer))

        tokenizer.save_pretrained(output_path)

        huggingface.train_mlm(docs, loaded_model, tokenizer, output_path, 2)

        self.assertTrue(os.path.exists(output_path) and os.listdir(output_path))
        shutil.rmtree(output_path)

    @mock.patch('huggingface.input')
    def test_get_new_tokens(self, mock_input):
        data = [
            "Ground spice commonly used in Indian cooking and drinks, in Middle Eastern cooking and in Scandinavian baking",
            "Alcoholic beverage produced by fermentation of grape juice produced from the chardonnay grape, with little contact with grape skins so that the wine is pale yellow in colour with little sugars remaining. Includes wine produced in Australia. Approximately 13% v/v alcohol."
        ]

        answer = [('Alcoholic', 1), ('Approximately', 2), ('Australia', 3), ('Eastern', 4), ('Ground', 5),
                  ('Includes', 6), ('Indian', 7), ('Middle', 8), ('Scandinavian', 9), ('chardonnay', 13),
                  ('fermentation', 19), ('sugars', 28)]

        model_name = "bert-base-uncased"

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        different_tokens_list = huggingface.get_new_tokens(data, tokenizer)

        self.assertEqual(different_tokens_list, answer)

    @mock.patch('huggingface.input')
    def test_spacy_tokenizer(self, mock_input):
        document = 'Ground spice commonly used in Indian cooking and drinks, in Middle Eastern cooking and in ' \
                   'Scandinavian baking '

        answer = ['Ground', 'spice', 'commonly', 'Indian', 'cooking', 'drinks', 'Middle', 'Eastern',
                  'cooking', 'Scandinavian', 'baking']

        tokens = huggingface.spacy_tokenizer(document)

        self.assertEqual(tokens, answer)

    def test_dfreq(self):
        length = 2
        idf_sorted = np.array([1.4054651081081644, 1.4054651081081644, 1.4054651081081644, 1.4054651081081644,
                               1.4054651081081644, 1.4054651081081644, 1.4054651081081644, 1.4054651081081644,
                               1.4054651081081644, 1.4054651081081644, 1.4054651081081644, 1.4054651081081644,
                               1.4054651081081644, 1.4054651081081644, 1.4054651081081644, 1.4054651081081644,
                               1.4054651081081644, 1.4054651081081644, 1.4054651081081644, 1.4054651081081644,
                               1.4054651081081644, 1.4054651081081644, 1.4054651081081644, 1.4054651081081644,
                               1.4054651081081644, 1.4054651081081644, 1.4054651081081644, 1.4054651081081644,
                               1.4054651081081644, 1.4054651081081644, 1.4054651081081644, 1.4054651081081644])
        answer = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        response = list(huggingface.dfreq(idf_sorted, length))

        self.assertEqual(response, answer)

    def test_error_message(self):
        message = "this is an error message test"
        response = huggingface.error_message(message, 500)
        data = json.loads(response.get_data(as_text=True))
        self.assertEqual(data["message"], message)
        self.assertEqual(response.status_code, 500)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(model_root)


if __name__ == '__main__':
    unittest.main()
