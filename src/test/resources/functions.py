import json

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
