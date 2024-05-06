
import nltk
import json
import pickle
import numpy as np
import random
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Initialize NLTK's WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents data
data_file = open('data.json').read()
intents = json.loads(data_file)

# Initialize empty lists and variables
words = []
classes = []
documents = []
ignore_words = ['?', '!']

# Process intents and patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        word_tokens = nltk.word_tokenize(pattern)
        words.extend(word_tokens)
        # Add documents in the corpus
        documents.append((word_tokens, intent['tag']))
        # Add to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lower each word, remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Save preprocessed data
pickle.dump(words, open('texts.pkl', 'wb'))
pickle.dump(classes, open('labels.pkl', 'wb'))

# Initialize training data
training = []
output_empty = [0] * len(classes)

# Create bag of words for each sentence in training set
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffle and convert training data into numpy array
random.shuffle(training)
training = np.array(training)
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Create the model
model = Sequential([
    Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(train_y[0]), activation='softmax')
])

# Compile the model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fit the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=500, batch_size=5, verbose=1)

# Save the model
model.save('model.h5', hist)

print("Model created")
