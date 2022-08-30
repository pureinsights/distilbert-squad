import json
from src.main.download import download_model, download_from_huggingface, download_models


def remove_scores_from_response(json_data):
    answer_response = []
    for data in json_data:
        del data["score"]
        answer_response.append(data)

    return answer_response


def get_post_response(tester, endpoint, body, content_type, convert_json=False):
    response = tester.post(
        endpoint,
        content_type=content_type,
        json=body
    )

    response_text = response.get_data(as_text=True)

    return json.loads(response_text) if convert_json else response_text, response.status_code


def get_download_model(path, model):
    output_path = path + "/" + model + "/"

    current_path, model_exist = download_model(path, model)
    same_path, same_model_exist = download_model(path, model)

    return current_path, model_exist, same_path, same_model_exist, output_path

