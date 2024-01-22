# views.py
from django.shortcuts import render
from django.http import HttpResponse
from keras.models import load_model
from keras.preprocessing import sequence
import joblib
import numpy as np

def sentiment_analysis_view(request):
    if request.method == 'POST':
        # Get the user input
        text = request.POST.get('text', '')

        # Load the saved tokenizer and one-hot encoder
        tokenizer, binr = joblib.load('data/sentiment_analysis_preprocessing.joblib')

        # Tokenize the text
        sequences = tokenizer.texts_to_sequences([text])
        x = sequence.pad_sequences(sequences, maxlen=50)

        # Load the Keras model
        model = load_model('data/sentiment_analysis_model.h5')

        # Perform the prediction
        prediction = model.predict(x)
        prediction = np.argmax(prediction, axis=1)[0]

        # Debugging: Print the prediction to the console
        print("Prediction:", prediction)

        # Render the template with the prediction result
        return render(request, 'sentiment_analysis.html', {'prediction': prediction})
    else:
        return render(request, 'sentiment_analysis.html')
