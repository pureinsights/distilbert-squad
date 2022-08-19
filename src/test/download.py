import shutil
import unittest
from sys import path
import os

path.insert(0, os.getcwd() + "/../")
from src.main.download import download_model, download_from_huggingface

model_root = "./models_test"
model_name = 'prajjwal1/bert-tiny'


class TestDownload(unittest.TestCase):

    def test_download_model(self):
        model = {"model": model_name}
        current_path = download_model(model_root, model)
        self.assertEqual(current_path, model_root + "/" + model_name + "/")
        shutil.rmtree(model_root)

    def test_download_from_huggingface(self):
        model_path = "{}/{}/".format(model_root, model_name)
        download_from_huggingface(model_name, model_path)
        self.assertTrue(os.path.exists(model_path) and os.listdir(model_path))
        shutil.rmtree(model_root)


if __name__ == '__main__':
    unittest.main()