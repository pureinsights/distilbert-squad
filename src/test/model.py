import shutil
import unittest
from sys import path
import os
from transformers import pipelines

path.insert(0, os.getcwd() + "/../")
from src.main.model import download_models, load_models, Model

model_root = "./models_test"
model_name = 'sshleifer/tiny-distilbert-base-cased-distilled-squad'


class TestModel(unittest.TestCase):

    def test_download_models(self):
        model = [{"model": model_name}]
        download_models(model, model_root)

        model_path = "{}/{}/".format(model_root, model_name)
        self.assertTrue(os.path.exists(model_path) and os.listdir(model_path))
        shutil.rmtree(model_root)

    def test_load_models(self):
        model = [{"model": model_name}]
        download_models(model, model_root)

        pipelines_, default_pipeline = load_models(model_root, -1)

        self.assertIsInstance(default_pipeline, pipelines.question_answering.QuestionAnsweringPipeline)
        self.assertIsInstance(pipelines_[model_name], pipelines.question_answering.QuestionAnsweringPipeline)
        shutil.rmtree(model_root)

    def test_model_get_pipeline(self):
        model = [{"model": model_name}]
        download_models(model, model_root)

        model_class = Model(model_root)
        self.assertIsInstance(model_class.get_pipeline(model_name),
                              pipelines.question_answering.QuestionAnsweringPipeline)
        shutil.rmtree(model_root)

    def test_model_predict(self):
        model = [{"model": 'deepset/roberta-base-squad2'}]
        download_models(model, model_root)

        texts = ['The FA Cup is open to any eligible club down to Level 10 of the English football league system – 20 '
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
        question = "How many games are required to win the FA Cup?"

        answer = [{'score': 0.6043325066566467, 'start': 693, 'end': 708, 'answer': 'six to fourteen'},
                  {'score': 0.0016566978301852942, 'start': 64, 'end': 66, 'answer': '32'}]

        model_class = Model(model_root)
        self.assertEqual(model_class.predict(texts, question, 'deepset/roberta-base-squad2'), answer)
        shutil.rmtree(model_root)


if __name__ == '__main__':
    unittest.main()
