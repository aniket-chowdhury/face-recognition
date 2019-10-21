from keras.models import load_model
from load_data import data

#input path
path = input("Path: ")

#model loading
model = load_model("models/"+path+"/model.h5")

#split data load
left,right,out = data(50)

# model prediction
run = model.predict([left,right])

print(run)

print(out)
