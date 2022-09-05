import shutil
import unittest
import os

from src.main.download import download_model, download_from_huggingface, download_models
from src.test.resources.constants import MODEL_ROOT, PATHS_TO_DOWNLOAD, MODEL_ROOT_QA, MODEL_ROOT_ST, BERT_TINY_MODEL, \
    TINY_DISTILBERT_MODEL, BERT_UNCASED_MODEL
from src.test.resources.functions import get_download_model


class TestDownload(unittest.TestCase):

    def test_download_model(self):
        model = BERT_TINY_MODEL

        current_path_qa, model_exist_qa, same_path_qa, same_model_exist_qa, output_path_qa = get_download_model(
            MODEL_ROOT_QA, model)
        current_path_st, model_exist_st, same_path_st, same_model_exist_st, output_path_st = get_download_model(
            MODEL_ROOT_ST, model)

        self.assertEqual(current_path_qa, output_path_qa)
        self.assertEqual(same_path_qa, output_path_qa)
        self.assertFalse(model_exist_qa)
        self.assertTrue(same_model_exist_qa)

        self.assertEqual(current_path_st, output_path_st)
        self.assertEqual(same_path_st, output_path_st)
        self.assertFalse(model_exist_st)
        self.assertTrue(same_model_exist_st)

        shutil.rmtree(MODEL_ROOT)

    def test_download_nonexistent_model(self):
        model = "nonexistent_model"
        self.assertRaises(OSError, lambda: download_model(MODEL_ROOT_QA, model))
        self.assertRaises(OSError, lambda: download_model(MODEL_ROOT_ST, model))

    def test_download_from_huggingface(self):
        model_path_qa = "{}/{}/".format(MODEL_ROOT_QA, BERT_TINY_MODEL)
        model_path_st = "{}/{}/".format(MODEL_ROOT_ST, BERT_TINY_MODEL)

        download_from_huggingface(BERT_TINY_MODEL, model_path_qa)
        download_from_huggingface(BERT_TINY_MODEL, model_path_st)

        self.assertTrue(os.path.exists(model_path_qa) and os.listdir(model_path_qa))
        self.assertTrue(os.path.exists(model_path_st) and os.listdir(model_path_st))

        shutil.rmtree(MODEL_ROOT)

    def test_download_models(self):
        models = [
            TINY_DISTILBERT_MODEL
        ]
        models_exist = [
            BERT_UNCASED_MODEL
        ]
        models_error = [
            TINY_DISTILBERT_MODEL,
            "noneexistant_model",
            BERT_UNCASED_MODEL
        ]

        answer1 = ("{'questionAndAnswer': [{'model': 'sshleifer/tiny-distilbert-base-cased-distilled-squad', "
                   "'path': './models_test/questionAndAnswer/sshleifer/tiny-distilbert-base-cased-distilled-squad/', "
                   "'status': 'Model downloaded'}]}", False)

        answer2 = ("{'sentenceTransformer': [{'model': 'sshleifer/tiny-distilbert-base-cased-distilled-squad', "
                   "'path': './models_test/sentenceTransformer/sshleifer/tiny-distilbert-base-cased-distilled-squad/', "
                   "'status': 'Model downloaded'}]}", False)

        answer3 = ("{'questionAndAnswer': [{'model': 'bert-base-uncased', 'path': "
                   "'./models_test/questionAndAnswer/bert-base-uncased/', 'status': 'Model downloaded'}]}", False)

        answer4 = ("{'sentenceTransformer': [{'model': 'bert-base-uncased', 'path': "
                   "'./models_test/sentenceTransformer/bert-base-uncased/', 'status': 'Model downloaded'}]}", False)

        answer5 = ("{'questionAndAnswer': [{'model': 'sshleifer/tiny-distilbert-base-cased-distilled-squad', "
                   "'path': './models_test/questionAndAnswer/sshleifer/tiny-distilbert-base-cased-distilled-squad/', "
                   "'status': 'Model exists'}, {'model': 'noneexistant_model', 'status': \"We couldn't connect to "
                   "'https://huggingface.co/' to load this model and it looks like noneexistant_model is not the path "
                   "to a directory conaining a config.json file.\\nCheckout your internet connection or see how to "
                   "run the library in offline mode at "
                   "'https://huggingface.co/docs/transformers/installation#offline-mode'.\"}, {'model': "
                   "'bert-base-uncased', 'path': './models_test/questionAndAnswer/bert-base-uncased/', 'status': "
                   "'Model exists'}]}",
                   True)

        answer6 = ("{'sentenceTransformer': [{'model': 'sshleifer/tiny-distilbert-base-cased-distilled-squad', "
                   "'path': './models_test/sentenceTransformer/sshleifer/tiny-distilbert-base-cased-distilled-squad/', "
                   "'status': 'Model exists'}, {'model': 'noneexistant_model', 'status': \"We couldn't connect to "
                   "'https://huggingface.co/' to load this model and it looks like noneexistant_model is not the path "
                   "to a directory conaining a config.json file.\\nCheckout your internet connection or see how to run "
                   "the library in offline mode at "
                   "'https://huggingface.co/docs/transformers/installation#offline-mode'.\"}, {'model': "
                   "'bert-base-uncased', 'path': './models_test/sentenceTransformer/bert-base-uncased/', 'status': "
                   "'Model exists'}]}",
                   True)

        models_downloaded1, is_error_found1 = download_models(PATHS_TO_DOWNLOAD, {"questionAndAnswer": models})
        models_downloaded2, is_error_found2 = download_models(PATHS_TO_DOWNLOAD, {"sentenceTransformer": models})

        models_downloaded3, is_error_found3 = download_models(PATHS_TO_DOWNLOAD, {"questionAndAnswer": models_exist})
        models_downloaded4, is_error_found4 = download_models(PATHS_TO_DOWNLOAD, {"sentenceTransformer": models_exist})

        models_downloaded5, is_error_found5 = download_models(PATHS_TO_DOWNLOAD, {"questionAndAnswer": models_error})
        models_downloaded6, is_error_found6 = download_models(PATHS_TO_DOWNLOAD, {"sentenceTransformer": models_error})

        self.assertTupleEqual((str(models_downloaded1), is_error_found1), answer1)
        self.assertTupleEqual((str(models_downloaded2), is_error_found2), answer2)
        self.assertTupleEqual((str(models_downloaded3), is_error_found3), answer3)
        self.assertTupleEqual((str(models_downloaded4), is_error_found4), answer4)
        self.assertTupleEqual((str(models_downloaded5), is_error_found5), answer5)
        self.assertTupleEqual((str(models_downloaded6), is_error_found6), answer6)

        shutil.rmtree(MODEL_ROOT)


if __name__ == '__main__':
    unittest.main()
