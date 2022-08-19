import shutil
import unittest
from sys import path
import os

path.insert(0, os.getcwd() + "/../")
from src.main.download import download_model, download_from_huggingface
from constants import MODEL_ROOT, BERT_TINY_MODEL


class TestDownload(unittest.TestCase):

    def test_download_model(self):
        model = {"model": BERT_TINY_MODEL}
        current_path = download_model(MODEL_ROOT, model)
        self.assertEqual(current_path, MODEL_ROOT + "/" + BERT_TINY_MODEL + "/")
        shutil.rmtree(MODEL_ROOT)

    def test_download_from_huggingface(self):
        model_path = "{}/{}/".format(MODEL_ROOT, BERT_TINY_MODEL)
        download_from_huggingface(BERT_TINY_MODEL, model_path)
        self.assertTrue(os.path.exists(model_path) and os.listdir(model_path))
        shutil.rmtree(MODEL_ROOT)


if __name__ == '__main__':
    unittest.main()
