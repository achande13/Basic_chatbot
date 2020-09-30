import tensorflow
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

with open("intents.json") as file:
    data = json.load(file)
#here we are trying to open the "trained" model first and if it is not trained then it will train the whole model and then continue to the chatbot
#this is basically done to avoid unnecessarily training the model again and again for each line we give as an input to the chatbot while using it
try:
  with open("data.pickle", "rb") as f: #rb is reading the "data.pickle" file in the form of bytes 
    words, label, t, output =pickle.load(f)
except:
  words=[]
  label=[]
  docx=[]
  docy=[]
  for intent in data["intents"]:
    for sample in intent["patterns"]:
      wrd= nltk.word_tokenize(sample) #this will return a list with all of the words in the pattern section of the dictionary
      words.extend(wrd)
      docx.append(wrd)
      docy.append(intent["tag"])

    if intend["tag"] not in label:
      label.append(intent["tag"])
  words = [stemmer.stem(w.lower()) for w in words if w != "?"] #the if statement is used to remove the question marks
  words= sorted(list(set(words))) #set will remove the duplicate data in the sorted list
  label = sorted(label) #sort all the labels
  #as neural network does not understand any words/ string but only numbers we need to make a bag of words
  #we are going to maintain the frequecy of all of our words that have been used which might be called as the bag of words
  #creating this bag of words
  t = [] #list for training
  output=[]

  out_mt= [0 for _ in range(len(label))]

  for x, doc in enumerate(docx):
    bag=[] #how many words we have
    word = [stmmer.stem(w) for w in doc]
    for w in words :
      if w in wrd:
        bag.append(1)
      else:
        bag.append(0)
    output_row = out_mt[:]
    output_row[label.index(docy[x])] = 1
    t.append(bag)
    output.append(output_row)
    with open("data.pickle", "wb") as f: #write all of these variables into the pickle file so that we can save it
      pickle.dump((words, label, t, output), f) 

  t= np.array(t)
  output = np.array(output)
#building the model using tflearn
tensorflow.reset_default_graph() #before doing this project, i might have used tensorflow as tf, so this is done to reset all the allocations and the settings to default
net= tflearn.input_data(shape= [None, len(t[0])])
net= tflearn.fully_connected(net, 8) #this is the first hidden layer of the neural network
net= tflearn.fully_connected(net, 8)
net= tflearn.fully_connected(net, len(output[0]), activation = "softmax" ) #softmax will give the probabillity of each neuron in the current hidden network
net= tflearn.regression(net)


model= tflearn.DNN(net) #DNN is the deep neural network with multiple hidden layers
#this will check if the model is trained already or not
try:
  model.load(model.tflearn)
except:
  model.fit(t, output, n_epoch=1000, batch= 8, show_metric=True)
  model.save("model.tflearn")

#now it is actually time to make prediction
def wordbag(s, words):
  bag=[0 for _ in range(len(words))] #here we are going to store all the words , same as done before
  s_words=nltk.word_tokenize(s)
  s_words=[stemmer.stem(word.lower()) for k in s_words ]

  for se in s_words:
    for i, u in enumerate(words):
      if u == se:
        bag[i] = 1
  return np.array(bag)

#ask the user for some kind of sentence and get some response
def chat():
  print("Start talking with the bot and use the word quit to end the session")
  inp= input("Y: ")
  if inp.lower() == "quit": #end the session once you type "quit"
    break
  result= model.predict([ wordbag(inp, words) ]) #this will give the probabilty of the considered accurate reply
  result_index= numpy.argmax(result) #this will give the index of the greatest value in the above probabilies which will further be used to give a reply
  tag= label[results_index] #this will give the tag of the expected intent
  #now to get a response, we will have to find that tag and give the response from the respective found tag
  for t in data["intents"]:
    if t["tag"]== tag:
      responses= t["responses"]
  #now we will select one of the random responses out of the give dataset in the form of json
  print(random.choice(responses))



#running the chatbot
chat()



