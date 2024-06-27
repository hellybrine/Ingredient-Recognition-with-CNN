from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import requests

app = Flask(__name__)

model = load_model('model/identifylegacy.h5')

# Hardcoded names for now
vegetable_names = {
    0: 'cabbage',
    1: 'lettuce',
    2: 'mango',
    3: 'onion',
    4: 'orange',
    5: 'paprika',
    6: 'pear',
    7: 'peas',
    8: 'pineapple',
    9: 'pomegranate',
    10: 'potato',
    11: 'raddish',
    12: 'soy beans',
    13: 'spinach',
    14: 'sweetcorn',
    15: 'sweetpotato',
    16: 'tomato',
    17: 'turnip',
    18: 'watermelon',
    19: 'capsicum',
    20: 'carrot',
    21: 'cauliflower',
    22: 'chilli pepper',
    23: 'corn',
    24: 'cucumber',
    25: 'eggplant',
    26: 'garlic',
    27: 'ginger',
    28: 'grapes',
    29: 'jalapeno',
    30: 'kiwi',
    31: 'lemon'
}

SPOONACULAR_API_KEY = 'XXXXXXXXXXXXXXXXXXX' # API Key
BASE_URL = 'https://api.spoonacular.com/recipes/complexSearch'


def get_recipes(vegetable):
    params = {
        'query': vegetable,
        'apiKey': SPOONACULAR_API_KEY,
        'number': 5
    }
    response = requests.get(BASE_URL, params=params)
    recipes = response.json()
    return recipes


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        image = load_img(file, target_size=(100, 100))
        image = img_to_array(image) / 255.0
        image = image.reshape((1, 100, 100, 3))
        prediction = model.predict(image)
        predicted_class_index = prediction.argmax()  # Get the index of the highest probability
        predicted_class = vegetable_names[predicted_class_index]  # Get the corresponding vegetable name
        recipes = get_recipes(predicted_class)
        return jsonify({'vegetable': predicted_class, 'recipes': recipes})


if __name__ == '__main__':
    app.run(debug=True)
