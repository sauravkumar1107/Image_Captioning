
# coding: utf-8

# In[1]:


from os import listdir
from pickle import dump
from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg19 import preprocess_input
from keras.models import Model


# In[2]:


def extract_features(path):
    model = VGG19()
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    print(model.summary())
    
    features = dict()
    for name in listdir(path):
        filename = path +'/' + name
        image = load_img(filename, target_size = (224,224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        image_id = name.split('.')[0]
        features[image_id] = feature
        print('>%s' % name)
        
    return features


# In[3]:


path = 'H:/datasets/flickr dataset/Flicker8k_Dataset'
features = extract_features(path)
print('features :', len(features))
dump(features, open('features.pkl', 'wb'))


# In[92]:


def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text
filename = 'H:/datasets/flickr dataset/Flickr8k_text/Flickr8k.token.txt'
doc = load_doc(filename)


# In[93]:


def load_descriptions(doc):
    mapping = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        if len(line) < 2:
            continue
        image_id, image_desc = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        image_desc = ' '.join(image_desc)
        if image_id not in mapping:
            mapping[image_id] = list()
            
        mapping[image_id].append(image_desc)
    return mapping

descriptions = load_descriptions(doc)
print('loaded : %d ' % len(descriptions))
                


# In[94]:


import string
def clean_descriptions(descriptions):
    table = str.maketrans('','', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [w.translate(table) for w in desc]
            desc = [word for word in desc if len(word)>1]
            desc = [word for word in desc if word.isalpha()]
            desc_list[i] = ' '.join(desc)
            
clean_descriptions(descriptions)


# In[95]:


def to_vocabulary(descriptions):
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc

vocabulary = to_vocabulary(descriptions)
print('vocabulary size ', len(vocabulary))


# In[96]:


def save_descriptions(descriptions, filename):
    lines = list()
    for key , desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key+' '+ desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

save_descriptions(descriptions, 'descriptions.txt')


# In[97]:


from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint


# In[98]:


def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    for line in doc.split('\n'):
        if len(line) < 1:
            continue
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)

def load_clean_descriptions(filename, dataset):
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]
        if image_id in dataset:
            if image_id not in descriptions:
                descriptions[image_id] = list()
            desc = 'startseq' + ' '.join(image_desc) + 'endseq'
            descriptions[image_id].append(desc)
    return descriptions

def load_photo_features(filename, dataset):
    all_features = load(open(filename, 'rb'))
    features = {k: all_features[k] for k in dataset}
    return features

def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)

def create_sequences(tokenizer, max_length, desc_list, photo):
    X1, X2, y = list(), list(), list()
    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen = max_length)[0]
            out_seq = to_categorical([out_seq], num_classes = vocab_size)[0]
            
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
    return array(X1), array(X2), array(y)

def define_model(vocab_size, max_length):
    input1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(input1)
    fe2 = Dense(256, activation = 'relu')(fe1)
    
    input2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(input2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation = 'relu')(decoder1)
    outputs = Dense(vocab_size, activation = 'softmax')(decoder2)
    
    model = Model(inputs=[input1, input2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model

def data_generator(descriptions, photos, tokenizer, max_length):
    while 1:
        for key, desc_list in descriptions.items():
            photo = photos[key][0]
            in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo)
            yield [[in_img, in_seq], out_word]
        


# In[99]:


filename = 'H:/datasets/flickr dataset/Flickr8k_text/Flickr_8k.trainImages.txt'


# In[100]:


train = load_set(filename)
print(len(train))
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print(len(train_descriptions))
train_features = load_photo_features('features.pkl', train)
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index)+1
max_length = max_length(train_descriptions)
print(max_length)


# In[52]:



model = define_model(vocab_size, max_length)
epochs = 20
steps = len(train_descriptions)
for i in range(epochs):
    generator = data_generator(train_descriptions, train_features, tokenizer, max_length,)
    model.fit_generator(generator, epochs=1, steps_per_epoch = steps, verbose=1)
    model.save('model_' + str(i) + '.h5')


# In[101]:


from numpy import argmax
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq '
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' '+word
        if word == ' endseq ':
            break
    return in_text

def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    for key, desc_list in descriptions.items():
        yhat = generate_desc(model, tokenizer, photos[key], max_length)
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())
    print('BLEU-1 : ', corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2 : ', corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3 : ', corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4 : ', corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
    


# In[102]:


filename = 'H:/datasets/flickr dataset/Flickr8k_text/Flickr_8k.testImages.txt'


# In[103]:


test = load_set(filename)
test_descriptions = load_clean_descriptions('descriptions.txt', test)
test_features = load_photo_features('features.pkl', test)

filename = 'model_2.h5'
model = load_model(filename)
evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)


# In[104]:


filename = 'H:/datasets/flickr dataset/Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))


# In[105]:


tokenizer = load(open('tokenizer.pkl', 'rb'))
max_length = 32
model = load_model('model_20.h5')


# In[86]:


max_length = 32


# In[106]:


photo = extract_features('example_image_caption')


# In[107]:


photo['example'].shape


# In[108]:


description = generate_desc(model, tokenizer, photo['example'], max_length)
print(description)

