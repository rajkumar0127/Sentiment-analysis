from flask import Flask, request, render_template
import tensorflow as tf
from transformers import BertTokenizer
import numpy as np

app = Flask(__name__)

# Model path
MODEL_PATH = 'lite_Sentiment_model.tflite'  # Update with the correct path

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def preprocess_text(text):
    # Tokenize and prepare the input text
    inputs = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        pad_to_max_length=True,
        return_tensors="tf"
    )

    return inputs

def model_predict(text, interpreter):
    # Preprocess the text
    inputs = preprocess_text(text)

    # Perform inference using the loaded TFLite model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], inputs["input_ids"])
    interpreter.set_tensor(input_details[1]['index'], inputs["attention_mask"])

    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # The output_data is a 2D array with shape (1, 2) where the first dimension is batch size
    # and the second dimension is the number of classes (2 in this case).

    # Get the predicted label (0 or 1) from the output tensor
    predicted_label = int(np.argmax(output_data))

    # Return the predicted label
    return predicted_label



# Initialize the TFLite interpreter
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get text from POST request
        text = request.form['text']

        # Make prediction
        predicted_label = model_predict(text, interpreter)

        # Determine sentiment
        sentiment = "Negative" if predicted_label == 0 else "Positive"

        return render_template('result.html', text=text, sentiment=sentiment)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
