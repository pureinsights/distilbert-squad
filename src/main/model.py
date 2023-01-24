import json
from pathlib import Path
from os import environ
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

from transformers import (AutoTokenizer, AutoModelForQuestionAnswering, pipeline)
from sentence_transformers import SentenceTransformer

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)


class Model:
    def __init__(self, path: str):
        PATH_ENV_VARIABLE = "MODELS_PATH"
        CPU_GPU_DEVICE_VARIABLE = "CPU_GPU_DEVICE"
        # If an environment variable with MODEL_PATH has been set, then use it.
        self.path = environ[PATH_ENV_VARIABLE] if environ.get(PATH_ENV_VARIABLE) is not None else path
        self.path += "/" + self.field_name
        '''
        Device ordinal for CPU/GPU support. 
        Setting this to -1 will leverage CPU, >=0 will run the model on the associated CUDA device id.
        See https://huggingface.co/transformers/v3.0.2/main_classes/pipelines.html
        '''
        self.device = int(environ[CPU_GPU_DEVICE_VARIABLE]) if environ.get(CPU_GPU_DEVICE_VARIABLE) is not None else -1
        logger.debug("_init_ Model class CPU:%s , model path: %s", self.device,self.path)

    def get_models_stored(self, response):
        field_name = self.field_name
        for pipeline_name in self.pipelines:
            if field_name in response:
                response[field_name].append(pipeline_name)
            else:
                response[field_name] = [pipeline_name]

        return response


class ModelQuestionAnswer(Model):

    def __init__(self, path: str):
        self.field_name = "questionAndAnswer"
        super().__init__(path)
        self.pipelines, self.default_pipeline = self.load_models()
        logger.debug("_init_ class:%s",self.field_name)

    def get_pipeline(self, model_name):
        """
        Chooses which pipeline to use.
        If no model_name is specific then use the default pipeline.
        if a model_name is provided but it doesn't exist, then return the default pipeline.
        @param model_name: Name of the model.
        @return: Pipeline associated to the model name.
        """
        if model_name is None or model_name not in self.pipelines.keys():
            logger.debug("Get default model questionAndAnswer")
            return self.default_pipeline
        else:
            logger.debug("Get default model:%s",model_name)
            return self.pipelines[model_name]

    def predict(self, contexts, question, model_name):
        """
        Locates the answer to a question across several context texts.
        @param contexts: Possible answers.
        @param question: Posed question.
        @param model_name: Model name to use.
        @return: Object which contains the answer, its position and a score.
        """
        logger.debug("Prediction in progress...")
        questions = [question] * len(contexts)  # There must be a context text per asked question
        selected_pipeline = self.get_pipeline(model_name)
        answers = selected_pipeline(question=questions, context=contexts)

        # if a single element is returned, then convert it to list
        if not isinstance(answers, list):
            answers = [answers]

        return answers

    def reload_models(self):
        self.pipelines, self.default_pipeline = self.load_models()

    def load_models(self):
        """
        Iterates over a specific path and loads all models available.
        For each model a pipeline is created and stored in a pipelines map,
        where the key is the model name (as defined in the config.json file)
        or the folder path if that value is not present.
        @return: Tuple consisting of: a map of pipelines and a default pipeline.
        """
        logger.debug("Loading QA models...")
        pipelines = {}
        default_pipeline = None
        config_file = 'config.json'
        for config_file_path in Path(self.path).rglob(config_file):
            logger.debug("Accesing config file")
            model_path = str(config_file_path.parent)

            name_path_key = '_name_or_path'
            with open(config_file_path) as jsonFile:
                json_object = json.load(jsonFile)
                # Looks for the file name in the config file. If not present then it will use the path as the name.
                model_name = json_object[name_path_key] if name_path_key in json_object else str(
                    config_file_path.parent).replace(self.path, '')
                logger.debug("Model name from config file:%s",model_name)

            question_answer_tokenizer = AutoTokenizer.from_pretrained(model_path)
            question_answer_model = AutoModelForQuestionAnswering.from_pretrained(model_path)

            current_pipeline = pipeline('question-answering',
                                        model=question_answer_model,
                                        tokenizer=question_answer_tokenizer,
                                        device=self.device)

            pipelines[model_name] = current_pipeline
            # Sets the first model found as the default one.
            if default_pipeline is None:
                logger.debug("Set default model")
                default_pipeline = current_pipeline
            logger.debug("Model loaded: %s", model_name)
        return pipelines, default_pipeline


class ModelSentenceTransformer(Model):

    def __init__(self, path: str):
        self.field_name = "sentenceTransformer"
        super().__init__(path)
        self.device = "cpu" if self.device == -1 else "cuda"
        self.pipelines, self.default_pipeline = self.load_models()
        logger.debug("_init_ class:%s",self.field_name)

    def get_pipeline(self, model_name):
        """
        Chooses which pipeline to use.
        If no model_name is specific then use the default pipeline.
        if a model_name is provided but it doesn't exist, then return the default pipeline.
        @param model_name: Name of the model.
        @return: Pipeline associated to the model name.
        """

        logger.debug("Get pipeline from sentence transformer")
        if model_name is None or model_name not in self.pipelines.keys():
            return self.default_pipeline
        else:
            return self.pipelines[model_name]

    def reload_models(self):
        self.pipelines, self.default_pipeline = self.load_models()

    def load_models(self):
        """
        Iterates over a specific path and loads all models available.
        For each model a pipeline is created and stored in a pipelines map,
        where the key is the model name (as defined in the config.json file)
        or the folder path if that value is not present.
        @return: Tuple consisting of: a map of pipelines and a default pipeline.
        """
        logger.debug("Loading Sentence Transformer models...")
        pipelines = {}
        default_pipeline = None
        config_file = 'config.json'
        for config_file_path in Path(self.path).rglob(config_file):
            logger.debug("Accesing config file")
            model_path = str(config_file_path.parent)

            name_path_key = '_name_or_path'
            with open(config_file_path) as jsonFile:
                json_object = json.load(jsonFile)
                # Looks for the file name in the config file. If not present then it will use the path as the name.
                model_name = json_object[name_path_key] if name_path_key in json_object else str(
                    config_file_path.parent).replace(self.path, '')
                try:
                    sentence_transformer_model = SentenceTransformer(model_path, device=self.device)
                    pipelines[model_name] = sentence_transformer_model
                    if default_pipeline is None:
                        default_pipeline = sentence_transformer_model
                    print(f"Model loaded: {model_name}")
                    logger.debug("Model loaded: %s", model_name)
                except:
                    print(f"Couldn't load model: {model_name}. Skipping it.")
            
        return pipelines, default_pipeline


    def encode(self, document_id, texts, model_name):
        """
        Encodes the texts.
        @param document_id: document id.
        @param texts: texts to be encoded.
        @param model_name: Model name to use.
        @return: An embedding and the id of the document.
        """
        logger.debug("Execute encode method")
        model = self.get_pipeline(model_name)
            
        if not model:
            model = SentenceTransformer(model_name, device=self.device)
            
        texts_encoded = model.encode(texts,batch_size=64, device=self.device)


        return {"id": document_id, "result": texts_encoded.tolist()}


