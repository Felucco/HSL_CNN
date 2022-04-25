from tensorflow.keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Activation
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import History
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.backend import function as Kfunction
import numpy as np
BATCH_SIZE = 128
EPOCHS = 15
LR = 0.009
TRAIN = False

def build_model(in_shape: np.ndarray) -> Sequential:
    model = Sequential()
    model.add(Conv2D(10,7,padding="same",kernel_initializer="glorot_normal",bias_initializer="glorot_normal",input_shape=in_shape)) #Using padding=same to emulate pre-padding + padding=valid; Glorot normal = Xavier
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=2,strides=2))
    model.add(Conv2D(20,7,padding="same",kernel_initializer="glorot_normal",bias_initializer="glorot_normal"))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=2,strides=2))
    model.add(Flatten())
    model.add(Dense(84))
    model.add(Activation("relu"))
    model.add(Dense(10)) #Linear activation as in the original model

    opt = Adam(learning_rate=LR)
    loss = CategoricalCrossentropy(from_logits=True) #As no softmax function is applied to the last layer

    model.compile(optimizer=opt,loss=loss,metrics=["accuracy"])
    return model

def fit_model(model: Sequential, X_train, y_train, X_test, y_test) -> History:
    checkpoint = ModelCheckpoint("best_model_w",save_best_only=True,save_weights_only=True)
    history = model.fit(X_train,y_train,batch_size=BATCH_SIZE,epochs=EPOCHS,callbacks=[checkpoint],validation_data=(X_test,y_test))
    return history

def save_ws(model: Sequential, output=str):
    with open("D:\\Documenti\\Università\\Embedded\\CNN-using-HLS\\modules\\fully_connected\\headers\\weights.h","w") as f:
        fc1 = model.get_layer("dense")
        biases = np.array(fc1.weights[1]).flatten()
        kernels = np.array(fc1.weights[0]).flatten()

        print("//AUTO_GENERATED WITH get_outputs.py",file=f,end="\n//--------------------------------------------\n")
        print('#ifndef __FC_WS_H\n#define __FC_WS_H\n\n#include "defines.h"\n\nfloat fc_bias[FC_BIAS_SIZE] = {',file=f,end="")

        for w in biases[:-1]:
            print(w,file=f,end=",")
        print(biases[-1],file=f,end="};\n")

        print('\n\nfloat fc_weights[FC_WEIGHTS_H][FC_WEIGHTS_W] = {',file=f,end="")
        
        for w in kernels[:-1]:
            print(w,file=f,end=",")
        print(kernels[-1],file=f,end="};\n")
        print("#endif",file=f)

def get_exp_outputs(model: Sequential, tests:np.ndarray):
    flat_layer = model.get_layer("flatten")
    flat_model = Model(inputs = model.input, outputs = flat_layer.output)
    dense_layer = model.get_layer("dense")
    dense_model = Model(inputs=dense_layer.input, outputs=model.get_layer(index=8).output)
    fc_out = open("fc_py.out","w")
    flat_out = open("flatten_py.out","w")
    
    flats = flat_model.predict(tests)
    denses = dense_model.predict(flats)

    for flat,dense in zip(flats,denses):
        for f in flat[:-1]:
            print(f,file=flat_out,end=" ")
        print(flat[-1],file=flat_out)

        for d in dense[:-1]:
            print(d,file=fc_out, end=" ")
        print(dense[-1],file=fc_out)
    
    fc_out.close()
    flat_out.close()
        

def main():
    global model
    (X_train,y_train),(X_test,y_test) = mnist.load_data()
    X_train = X_train.astype("float32") / 255.0
    X_train = X_train.reshape((-1,X_train.shape[1],X_train.shape[2],1))
    X_test = X_test.astype("float32") / 255.0
    X_test = X_test.reshape((-1,X_test.shape[1],X_test.shape[2],1))
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    model = build_model(X_train.shape[1:])
    if TRAIN:
        fit_model(model,X_train,y_train,X_test,y_test)
    model.load_weights("best_model_w")
    save_ws(model)
    get_exp_outputs(model,X_test[:10])

if __name__ == "__main__":
    main()