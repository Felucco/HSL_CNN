from tensorflow.keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Activation, ZeroPadding2D
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
TRAIN = True

def build_model(in_shape: np.ndarray) -> Sequential:
    model = Sequential()

    model.add(ZeroPadding2D(padding=2, input_shape=in_shape))

    model.add(Conv2D(8, kernel_size=4, strides=1, padding="valid", kernel_initializer="glorot_normal",
              bias_initializer="glorot_normal"))  # Glorot normal = Xavier
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=2, strides=2))

    model.add(Conv2D(16, kernel_size=2, padding="valid",
              kernel_initializer="glorot_normal", bias_initializer="glorot_normal"))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=2, strides=2))

    model.add(Flatten())

    model.add(Dense(120))
    model.add(Activation("relu"))

    model.add(Dense(84))
    model.add(Activation("relu"))

    model.add(Dense(10))  # Linear activation as in the original model

    opt = Adam(learning_rate=LR)
    # As no softmax function is applied to the last layer
    loss = CategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    return model

def fit_model(model: Sequential, X_train, y_train, X_test, y_test) -> History:
    checkpoint = ModelCheckpoint("best_model_w",save_best_only=True,save_weights_only=True)
    history = model.fit(X_train,y_train,batch_size=BATCH_SIZE,epochs=EPOCHS,callbacks=[checkpoint],validation_data=(X_test,y_test))
    return history

def save_ws(model: Sequential, output=str):
    with open("D:\\Documenti\\Università\\Embedded\\CNN-using-HLS\\nnet_stream\\new\\modules\\conv\\headers\\weights.h","w") as f:
        conv1 = model.get_layer(index=1)
        biases = np.array(conv1.weights[1]).flatten()
        kernels = np.array(conv1.weights[0]).flatten()

        print("//AUTO_GENERATED WITH get_outputs.py",file=f,end="\n//--------------------------------------------\n")
        print('#ifndef __CONV1_WS_H\n#define __CONV1_WS_H\n',file=f)
        with open("D:\\Documenti\\Università\\Embedded\\CNN-using-HLS\\nnet_stream\\new\\modules\\conv\\headers\\f24t_base") as f24t:
            print(f24t.read(),file=f,end="\n\n")
        print('#include "defines.h"\n\nfloat fc_bias[FC_BIAS_SIZE] = {',file=f,end="")

        print(
            f"\nfloat24_t conv_bias[{biases.shape[0]}] ="+"{ ", file=f, end="")
        for w in biases[:-1]:
            print(w, file=f, end=",")
        print(biases[-1], file=f, end="};\n")

        shape_str = "".join([f"[{sh}]" for sh in kernels.shape])
        print(
            f"\nfloat24_t conv_weights{shape_str} ="+"{ ", file=f, end="")
        k_flat = kernels.flatten()
        for w in k_flat[:-1]:
            print(w, file=f, end=",")
        print(k_flat[-1], file=f, end="};\n")
        print("#endif",file=f)

def get_exp_outputs(model: Sequential, tests:np.ndarray):
    pad_layer = model.get_layer(index=0)
    pad_model = Model(inputs = model.input, outputs = pad_layer.output)
    conv_layer = model.get_layer(index=1)
    conv_model = Model(inputs=conv_layer.input, outputs=model.get_layer(index=2).output) #Activation included
    conv_out = open("conv_py.out","w")
    pad_out = open("pad_py.out","w")
    
    pads = pad_model.predict(tests)
    convs = conv_model.predict(pads)

    for pad,conv in zip(pads,convs):
        pad_flat = np.array(pad).flatten()
        for f in pad_flat[:-1]:
            print(f,file=pad_out,end=" ")
        print(pad_flat[-1],file=pad_out)

        conv_flat = np.array(conv).flatten()
        for d in conv_flat[:-1]:
            print(d,file=conv_out, end=" ")
        print(conv_flat[-1],file=conv_out)
    
    conv_out.close()
    pad_out.close()
        

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