<!-- Space: PDP -->
<!-- Parent: User Documentation -->
<!-- Parent: Distilbert Squad -->
<!-- Title: Distilbert Squad -->

# PDP Distilbert 

This repository contains code related to NLP tasks.

## Download Models

### Hugging Face Models

First and foremost, you need to download models locally. For this, `init_models.py` can be executed. In here you can configure the models to download and a path where they will be stored:

```python
models = [{"model": 'deepset/roberta-base-squad2'},
          {"model": 'oliverproud/distilbert-finetuned-model'},
          {"model": 'mrm8488/distilbert-multi-finetuned-for-xqua-on-tydiqa'},
          {"model": 'distilbert-base-cased-distilled-squad'}]
path = "./models"
```
THe model name needs to match the name in the [Hugging Face repository](https://huggingface.co/models).

### Custom Models

Custom models can be installed too, however this needs to be done manually. You will need to copy your model to the expected path (as defined in variable `path`).


# Endpoints 

## Predict

Given a question and one or various texts, extract from each text the section containing the answer to the question.

Verb: `POST`

Endpoint: `http://localhost:8080/predict`

Payload: 
```json
{
    "model": "distilbert-base-cased-distilled-squad",
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
```

## Parameters:

`model` - Optional, string.

Model to be used to make the prediction, to see the available options use the models endpoint.

`question` - Required, string.

Question to ask the model

`chunks` - Required, node list.

List of nodes containing a text and an id (both string), an answer will be extracted (if possible) from every text in the chunk


Sample Response:
```json
[
    {
        "score": 0.5027955770492554,
        "start": 693,
        "end": 708,
        "answer": "six to fourteen",
        "id": "1",
        "highlight": "The FA Cup is open to any eligible club down to Level 10 of the English football league system – 20 professional clubs in the Premier League (level 1),72 professional clubs in the English Football League (levels 2 to 4), and several hundred non-League teams in steps 1 to 6 of the National League System (levels 5 to 10). A record 763 clubs competed in 2011–12. The tournament consists of 12 randomly drawn rounds followed by the semi-finals and the final. Entrants are not seeded, although a system of byes based on league level ensures higher ranked teams enter in later rounds.  The minimum number of games needed to win, depending on which round a team enters the competition, ranges from <span class=\"highlight\">six to fourteen</span>."
    },
    {
        "score": 0.15806056559085846,
        "start": 10,
        "end": 13,
        "answer": "six",
        "id": "2",
        "highlight": "The first <span class=\"highlight\">six</span> rounds are the Qualifying Competition, from which 32 teams progress to the first round of the Competition Proper, meeting the first of the 48 professional teams from Leagues One and Two. The last entrants are the Premier League and Championship clubs, into the draw for the Third Round Proper.[2] In the modern era, only one non-League team has ever reached the quarter-finals, and teams below Level 2 have never reached the final.[note 1] As a result, significant focus is given to the smaller teams who progress furthest, especially if they achieve an unlikely \"giant-killing\" victory."
    }
]
```

## Models

Get all the models that can be found at the .models path

Verb: `GET`

Endpoint: `http://localhost:8080/models`

Payload: None

Sample Result:
```json
[
    "{path}/deepset/roberta-base-squad2",
    "{path}/distilbert-base-cased-distilled-squad",
    "{path}/mrm8488/distilbert-multi-finetuned-for-xqua-on-tydiqa",
    "{path}/models\\oliverproud\\distilbert-finetuned-model"
]
```

## Train

Adds new vocabulary to a model based on the given data and trains it on masked language modeling.

Verb: `POST`

Endpoint: `http://localhost:8080/train`

Payload:
```json
{
  "output_path": "C:\\Users\\AdrianaMorales\\Desktop\\test-old",
  "model": "C:\\Users\\AdrianaMorales\\Desktop\\PDP-621",
  "data": [...],
  "batch_training": 2
}
```

## Parameters:

`output_path` - Required, string.

Path where the model will be saved

`model` - Required, string.

Name of the hugging face model that will be used for training, names can be found at the [Hugging face repository](https://huggingface.co/models).
A path to a model located at disk can also be provided in this field.

`data` - Required, string list.

Field name where output data will be placed.

`batch_training` - Optional, integer. Default is 2

Size of batch that will be used during training

Sample Result:

```json
{
    "message": "Training started",
    "timestamp": "2022-05-12T12:05:49.651371"
}
```

## Status

Indicates if the service is currently training a model.

Verb: `GET`

Endpoint: `http://localhost:8080/status`

Payload: None

Sample Result:
```json
{
  "training_status": true
}
```

## Download Models

Given valid and public model names, download and store them on disk.

Verb: `POST`

Endpoint: `http://localhost:8080/download-model`

Payload: 
```json
{
    "sentenceTransformer": [
        "bert-base-uncased"
    ],
    "questionAndAnswer": [
        "sshleifer/tiny-distilbert-base-cased-distilled-squad",
        "noneexistant_model",
        "bert-base-uncased"
    ]
}
```

## Parameters:

`sentenceTransformer` - Optional, list.

Models to search and download on disk for the specified transformer.

`questionAndAnswer` - Optional, list.

Models to search and download on disk for the specified transformer.

Sample Response:

Example 1
```json
{
    "message": {
        "questionAndAnswer": [
            {
                "model": "sshleifer/tiny-distilbert-base-cased-distilled-squad",
                "path": "../models/qa/sshleifer/tiny-distilbert-base-cased-distilled-squad/",
                "status": "Model downloaded"
            },
            {
                "model": "no_valid_model",
                "status": "We couldn't connect to 'https://huggingface.co/' to load this model and it looks like no_valid_model is not the path to a directory conaining a config.json file.\nCheckout your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'."
            },
            {
                "model": "bert-base-uncased",
                "path": "../models/qa/sshleifer/tiny-distilbert-base-cased-distilled-squad//bert-base-uncased/",
                "status": "Model downloaded"
            }
        ],
      "sentenceTransformer": [
          {
                "model": "bert-base-uncased",
                "path": "../models/st/sshleifer/tiny-distilbert-base-cased-distilled-squad//bert-base-uncased/",
                "status": "Model downloaded"
            }
        ]
    },
    "status": 400
}
```

Example 2
```json
{
  "questionAndAnswer": [
      {
        "model": "sshleifer/tiny-distilbert-base-cased-distilled-squad",
        "path": "../models/qa/sshleifer/tiny-distilbert-base-cased-distilled-squad/",
        "status": "Model downloaded"
      },
      {
        "model": "bert-base-uncased",
        "path": "../models/qa/sshleifer/tiny-distilbert-base-cased-distilled-squad//bert-base-uncased/",
        "status": "Model downloaded"
      }
    ],
  "sentenceTransformer": [
    {
        "model": "bert-base-uncased",
        "path": "../models/st/sshleifer/tiny-distilbert-base-cased-distilled-squad//bert-base-uncased/",
        "status": "Model downloaded"
    }
  ]
}
```