import shutil
import unittest
from unittest import mock

import numpy as np
from flask import Flask
import os
import json
import spacy

import src.main.huggingface as huggingface
from src.main.model import ModelQA, ModelST
from src.main.download import download_models

from src.test.resources.constants import MODELS_QA, MODEL_ROOT, PATHS_TO_DOWNLOAD, MODEL_ROOT_TRAINED, \
    BERT_UNCASED_MODEL, \
    TINY_DISTILBERT_MODEL, PREDICT_ENDPOINT, CONTENTTYPE, DATA_TEST, TRAIN_ENDPOINT, DOWNLOAD_MODEL_ENDPOINT, MODELS_ST, \
    DATA_TEST_SINGLE_TEXT, ENCODE_ENDPOINT

from src.test.resources.functions import remove_scores_from_response, get_post_response

from transformers import AutoModelForMaskedLM, AutoTokenizer

app = Flask(__name__)
app.register_blueprint(huggingface.huggingface_api)


class TestEndpoint(unittest.TestCase):
    tester = None

    def __init__(self, *args, **kwargs):
        super(TestEndpoint, self).__init__(*args, **kwargs)
        global tester
        tester = app.test_client()

    @classmethod
    def setUpClass(cls):
        download_models(PATHS_TO_DOWNLOAD, MODELS_QA)
        download_models(PATHS_TO_DOWNLOAD, MODELS_ST)

    def setUp(self):
        huggingface.modelQA = ModelQA(MODEL_ROOT)
        huggingface.modelST = ModelST(MODEL_ROOT)

    @mock.patch('huggingface_test.input')
    def test_models(self, mock_input):
        response = tester.get(
            '/models',
            content_type=CONTENTTYPE
        )

        data = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data, '["./models_test/qa/deepset/roberta-base-squad2", '
                               '"./models_test/st/deepset/roberta-base-squad2"]')

    @mock.patch('huggingface_test.input')
    def test_predict(self, mock_input):
        body = {
            "model": TINY_DISTILBERT_MODEL,
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

        answer = "[{'start': 693, 'end': 708, 'answer': 'six to fourteen', 'id': '1', 'highlight': 'The FA Cup is " \
                 "open to any eligible club down to Level 10 of the English football league system – 20 professional " \
                 "clubs in the Premier League (level 1),72 professional clubs in the English Football League (levels " \
                 "2 to 4), and several hundred non-League teams in steps 1 to 6 of the National League System (levels " \
                 "5 to 10). A record 763 clubs competed in 2011–12. The tournament consists of 12 randomly drawn " \
                 "rounds followed by the semi-finals and the final. Entrants are not seeded, although a system of " \
                 "byes based on league level ensures higher ranked teams enter in later rounds.  The minimum number " \
                 "of games needed to win, depending on which round a team enters the competition, ranges from <span " \
                 "class=\"highlight\">six to fourteen</span>.'}, {'start': 64, 'end': 66, 'answer': '32', 'id': '2', " \
                 "'highlight': 'The first six rounds are the Qualifying Competition, from which <span " \
                 "class=\"highlight\">32</span> teams progress to the first round of the Competition Proper, " \
                 "meeting the first of the 48 professional teams from Leagues One and Two. The last entrants are the " \
                 "Premier League and Championship clubs, into the draw for the Third Round Proper.[2] In the modern " \
                 "era, only one non-League team has ever reached the quarter-finals, and teams below Level 2 have " \
                 "never reached the final.[note 1] As a result, significant focus is given to the smaller teams who " \
                 "progress furthest, especially if they achieve an unlikely \"giant-killing\" victory.'}]"

        response1, status_code1 = get_post_response(tester, PREDICT_ENDPOINT, body, CONTENTTYPE, convert_json=True)
        answer_response = remove_scores_from_response(response1)

        self.assertEqual(status_code1, 200)
        self.assertEqual(str(answer_response), answer)

    def test_predict_error_message(self):
        body_no_question = {
            "model": TINY_DISTILBERT_MODEL,
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
            "model": TINY_DISTILBERT_MODEL,
            "question": "How many games are required to win the FA Cup?"
        }

        _, status_code1 = get_post_response(tester, PREDICT_ENDPOINT, None, CONTENTTYPE)
        response2, status_code2 = get_post_response(tester, PREDICT_ENDPOINT, body_no_question, CONTENTTYPE, convert_json=True)
        response3, status_code3 = get_post_response(tester, PREDICT_ENDPOINT, body_no_chunks, CONTENTTYPE, convert_json=True)

        self.assertEqual(status_code1, 400)

        self.assertEqual(response2["message"], 'The prediction needs a question')
        self.assertEqual(status_code2, 400)

        self.assertEqual(response3["message"], 'The prediction needs at least one chunk')
        self.assertEqual(status_code3, 400)

    def test_status_with_no_train(self):
        response = tester.get(
            '/status',
            content_type=CONTENTTYPE
        )

        data = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data, '{"training_status": false}')

    def test_train(self):
        body = {
            "output_path": MODEL_ROOT_TRAINED,
            "model": BERT_UNCASED_MODEL,
            "data": DATA_TEST,
            "batch_training": 2
        }

        response1, status_code1 = get_post_response(tester, TRAIN_ENDPOINT, body, CONTENTTYPE, convert_json=True)

        self.assertEqual(status_code1, 200)
        self.assertEqual(response1["message"], "Training started")

    def test_train_error_message(self):
        body_no_output_path = {
            "model": BERT_UNCASED_MODEL,
            "data": DATA_TEST,
            "batch_training": 2
        }

        body_no_model = {
            "output_path": MODEL_ROOT_TRAINED,
            "data": DATA_TEST,
            "batch_training": 2
        }

        body_no_data = {
            "output_path": MODEL_ROOT_TRAINED,
            "model": BERT_UNCASED_MODEL,
            "batch_training": 2
        }

        body_data_empty = {
            "output_path": MODEL_ROOT_TRAINED,
            "model": BERT_UNCASED_MODEL,
            "data": None,
            "batch_training": 2
        }

        _, status_code1 = get_post_response(tester, TRAIN_ENDPOINT, None, CONTENTTYPE)
        response2, status_code2 = get_post_response(tester, TRAIN_ENDPOINT, body_no_output_path, CONTENTTYPE, convert_json=True)
        response3, status_code3 = get_post_response(tester, TRAIN_ENDPOINT, body_no_model, CONTENTTYPE, convert_json=True)
        response4, status_code4 = get_post_response(tester, TRAIN_ENDPOINT, body_no_data, CONTENTTYPE, convert_json=True)
        response5, status_code5 = get_post_response(tester, TRAIN_ENDPOINT, body_data_empty, CONTENTTYPE, convert_json=True)

        self.assertEqual(status_code1, 400)

        self.assertEqual(response2["message"], 'Missing parameter output_path')
        self.assertEqual(status_code2, 400)

        self.assertEqual(response3["message"], 'Missing parameter model')
        self.assertEqual(status_code3, 400)

        self.assertEqual(response4["message"], 'Missing parameter data')
        self.assertEqual(status_code4, 400)

        self.assertEqual(response5["message"], 'Missing information in parameter data')
        self.assertEqual(status_code5, 400)

    @mock.patch('huggingface_test.input')
    def test_download_models(self, mock_input):
        body1 = {
            "questionAndAnswer": [TINY_DISTILBERT_MODEL]
        }

        answer1 = '{"questionAndAnswer": [{"model": "sshleifer/tiny-distilbert-base-cased-distilled-squad", ' \
                  '"path": "./models_test/qa/sshleifer/tiny-distilbert-base-cased-distilled-squad/", "status": "Model ' \
                  'downloaded"}]}'

        body2 = {
            "questionAndAnswer": [
                TINY_DISTILBERT_MODEL,
                BERT_UNCASED_MODEL
            ]
        }

        answer2 = '{"questionAndAnswer": [{"model": "sshleifer/tiny-distilbert-base-cased-distilled-squad", ' \
                  '"path": "./models_test/qa/sshleifer/tiny-distilbert-base-cased-distilled-squad/", "status": "Model ' \
                  'exists"}, {"model": "bert-base-uncased", "path": "./models_test/qa/bert-base-uncased/", ' \
                  '"status": "Model downloaded"}]}'

        body3 = {
            "questionAndAnswer": [
                TINY_DISTILBERT_MODEL,
                "noneexistant_model",
                BERT_UNCASED_MODEL
            ]
        }

        answer3 = "{'message': {'questionAndAnswer': [{'model': " \
                  "'sshleifer/tiny-distilbert-base-cased-distilled-squad', 'path': " \
                  "'./models_test/qa/sshleifer/tiny-distilbert-base-cased-distilled-squad/', 'status': 'Model " \
                  "exists'}, {'model': 'noneexistant_model', 'status': \"We couldn't connect to " \
                  "'https://huggingface.co/' to load this model and it looks like noneexistant_model is not the path " \
                  "to a directory conaining a config.json file.\\nCheckout your internet connection or see how to run " \
                  "the library in offline mode at " \
                  "'https://huggingface.co/docs/transformers/installation#offline-mode'.\"}, {'model': " \
                  "'bert-base-uncased', 'path': './models_test/qa/bert-base-uncased/', 'status': 'Model exists'}]}, " \
                  "'status': 400}"

        response1, status_code1 = get_post_response(tester, DOWNLOAD_MODEL_ENDPOINT, body1, CONTENTTYPE)
        response2, status_code2 = get_post_response(tester, DOWNLOAD_MODEL_ENDPOINT, body2, CONTENTTYPE)
        response3, status_code3 = get_post_response(tester, DOWNLOAD_MODEL_ENDPOINT, body3, CONTENTTYPE, convert_json=True)
        del response3["timestamp"]

        self.assertEqual(status_code1, 200)
        self.assertEqual(response1, answer1)

        self.assertEqual(status_code2, 200)
        self.assertEqual(response2, answer2)

        self.assertEqual(status_code3, 400)
        self.assertEqual(str(response3), answer3)

        shutil.rmtree('./models_test/qa/sshleifer')
        shutil.rmtree('./models_test/qa/bert-base-uncased')

    @mock.patch('huggingface_test.input')
    def test_encode(self, mock_input):
        body = {
            "model": TINY_DISTILBERT_MODEL,
            "id": "123",
            "texts": [DATA_TEST_SINGLE_TEXT]
        }

        answer = "{'id': '123', 'result': [[0.007092174142599106, -0.007092198356986046]]}"

        response1, status_code1 = get_post_response(tester, ENCODE_ENDPOINT, body, CONTENTTYPE, convert_json=True)

        self.assertEqual(status_code1, 200)
        self.assertEqual(str(response1), answer)

    def test_encode_error_message(self):
        _, status_code1 = get_post_response(tester, ENCODE_ENDPOINT, None, CONTENTTYPE)

        self.assertEqual(status_code1, 400)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(MODEL_ROOT)


class TestEndpointFunctions(unittest.TestCase):
    tester = None

    def __init__(self, *args, **kwargs):
        super(TestEndpointFunctions, self).__init__(*args, **kwargs)
        global tester
        tester = app.test_client()

    @classmethod
    def setUpClass(cls):
        download_models(PATHS_TO_DOWNLOAD, MODELS_QA)

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
        model_name = BERT_UNCASED_MODEL
        output_path = MODEL_ROOT_TRAINED
        batch_size = 2

        answer = {"message": "MLM Training finished, model saved at: './models_test/trained/'",
                  "added_tokens": ["Alcoholic", "Approximately", "Australia", "Eastern", "Ground", "Includes", "Indian",
                                   "Middle", "Scandinavian", "chardonnay", "fermentation", "sugars"]}

        response = huggingface.start_train(model_name, output_path, batch_size, DATA_TEST)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        del data["timestamp"]
        self.assertEqual(data, answer)
        self.assertTrue(os.path.exists(MODEL_ROOT_TRAINED) and os.listdir(MODEL_ROOT_TRAINED))

        shutil.rmtree(MODEL_ROOT_TRAINED)

    def test_train_mlm(self):
        model_name = BERT_UNCASED_MODEL
        output_path = MODEL_ROOT_TRAINED

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        tokenizer.add_tokens(
            ['Alcoholic', 'Approximately', 'Australia', 'Eastern', 'Ground', 'Includes', 'Indian', 'Middle',
             'Scandinavian', 'chardonnay', 'fermentation', 'sugars'])

        loaded_model = AutoModelForMaskedLM.from_pretrained(model_name)
        loaded_model = loaded_model.to("cpu")
        loaded_model.resize_token_embeddings(len(tokenizer))

        tokenizer.save_pretrained(output_path)

        huggingface.train_mlm(DATA_TEST, loaded_model, tokenizer, output_path, 2)

        self.assertTrue(os.path.exists(output_path) and os.listdir(output_path))
        shutil.rmtree(output_path)

    @mock.patch('huggingface_test.input')
    def test_get_new_tokens(self, mock_input):
        answer = [('Alcoholic', 1), ('Approximately', 2), ('Australia', 3), ('Eastern', 4), ('Ground', 5),
                  ('Includes', 6), ('Indian', 7), ('Middle', 8), ('Scandinavian', 9), ('chardonnay', 13),
                  ('fermentation', 19), ('sugars', 28)]

        model_name = BERT_UNCASED_MODEL

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        different_tokens_list = huggingface.get_new_tokens(DATA_TEST, tokenizer)

        self.assertEqual(different_tokens_list, answer)

    @mock.patch('huggingface_test.input')
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
        shutil.rmtree(MODEL_ROOT)


if __name__ == '__main__':
    unittest.main()
