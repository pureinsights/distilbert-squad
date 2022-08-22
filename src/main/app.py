from flask import Flask
from src.main.huggingface import huggingface_api
import os

app = Flask(__name__)
app.register_blueprint(huggingface_api)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
