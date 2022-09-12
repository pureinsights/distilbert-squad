import os

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModel


def download_model(path, model_name):
    """
    Downloads models from Hugging Face is the 'download' flag has been set to true.
    If a folder with the model name already exists, then takes no action.
    Otherwise, the model should exist on disk.
    @param path: Path to where the models will be downloaded.
    @param model_name: Model name.
    @return: Path to the model on disk and if the model already exists
    @raise: throws an error if the model can't be downloaded and remove dir
    """
    model_path = "{}/{}/".format(path, model_name)
    model_exists = False
    try:
        if not os.path.isdir(model_path):
            os.makedirs(model_path)  # Create the dir
            print("Downloading model {} to {}".format(model_name, model_path))
            download_from_huggingface(model_name, model_path)
        else:
            print("Model already exists: {}".format(model_path))
            model_exists = True
    except Exception as e:
        os.rmdir(model_path)
        raise e

    return model_path, model_exists


def download_from_huggingface_question_answers(model_name, model_path):
    """
    To download files related to the model and the tokenizer, first it is needed to instantiate the model and the tokenizer,
    which will download the files from Hugging Face's repo. Then these are saved to disk.
    @param model_name: Name of the model (as expected to be found in Hugging Face)
    @param model_path: Path to where to store the model on disk
    @return:
    """
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model.save_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(model_path)


def download_from_huggingface_sentence_transformers(model_name, model_path):
    """
    To download files related to the model and the tokenizer, first it is needed to instantiate the model and the tokenizer,
    which will download the files from Hugging Face's repo. Then these are saved to disk.
    @param model_name: Name of the model (as expected to be found in Hugging Face)
    @param model_path: Path to where to store the model on disk
    @return:
    """
    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(model_path)


def download_models(paths, models):
    """
    Download models from Hugging Face repositories into a specific folder.
    If a folder already exists for a model then it is assumed it was downloaded before.
    @param paths: Models locations on disk.
    @param models: JSON objects with models to download.
    @return:the model name with its status and if an error was found
    """
    models_downloaded = {}
    is_error_found = False

    for model_type, model_names in models.items():
        models_downloaded.update({model_type: []})
        model_type_lower = model_type.lower()
        for model_name in model_names:
            try:
                path = paths[model_type_lower] if model_type_lower in paths else paths["default"]
                current_path, model_exists = download_model(path, model_name)
                models_downloaded[model_type].append(
                    {"model": model_name, "path": current_path,
                     "status": "Model exists" if model_exists else "Model downloaded"})
            except Exception as e:
                models_downloaded[model_type].append({"model": model_name, "status": str(e)})
                is_error_found = True

    return models_downloaded, is_error_found


def download_from_huggingface(model_name, model_path):
    """
    To download files related to the model and the tokenizer, first it is needed to instantiate the model and the tokenizer,
    which will download the files from Hugging Face's repo. Then these are saved to disk.
    @param model_path: Path to where to store the model on disk
    @param model_name: model name to be downloaded
    @return:
    """
    if "/st" in model_path:
        download_from_huggingface_sentence_transformers(model_name, model_path)
    else:
        download_from_huggingface_question_answers(model_name, model_path)
