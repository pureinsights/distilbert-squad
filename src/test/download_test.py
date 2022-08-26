import shutil
import unittest
import os

from src.main.download import download_model, download_from_huggingface_qa
from src.test.resources.constants import MODEL_ROOT, BERT_TINY_MODEL


class TestDownload(unittest.TestCase):

    def test_download_model(self):
        model = {"model": BERT_TINY_MODEL}
        output_path = MODEL_ROOT + "/" + BERT_TINY_MODEL + "/"

        current_path, model_exist = download_model(MODEL_ROOT, model)
        same_path, same_model_exist = download_model(MODEL_ROOT, model)

        self.assertEqual(current_path, output_path)
        self.assertEqual(same_path, output_path)
        self.assertFalse(model_exist)
        self.assertTrue(same_model_exist)

        shutil.rmtree(MODEL_ROOT)

    def test_download_nonexistent_model(self):
        model = {"model": "nonexistent_model"}
        self.assertRaises(OSError, lambda: download_model(MODEL_ROOT, model))

    def test_download_from_huggingface(self):
        model_path = "{}/{}/".format(MODEL_ROOT, BERT_TINY_MODEL)
        download_from_huggingface_qa(BERT_TINY_MODEL, model_path)
        self.assertTrue(os.path.exists(model_path) and os.listdir(model_path))
        shutil.rmtree(MODEL_ROOT)

    def test_download_from_huggingface_nonexistent_model(self):
        model_name = "nonexistent_model"
        model_path = "{}/{}/".format(MODEL_ROOT, model_name)
        self.assertRaises(OSError, lambda: download_from_huggingface_qa(model_name, model_path))


if __name__ == '__main__':
    unittest.main()
