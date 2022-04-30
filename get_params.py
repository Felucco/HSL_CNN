from typing import IO
from tensorflow.keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Activation, ZeroPadding2D, Layer
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import History
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np

WEIGHT_FILE = "headers/weights.h"
REF_FOLDER = "refs"

NUM_TESTS = 10

BATCH_SIZE = 128
EPOCHS = 20
LR = 0.009
TRAIN = False


def build_model(in_shape: np.ndarray) -> Sequential:
    model = Sequential()

    model.add(ZeroPadding2D(padding=2, input_shape=in_shape))

    model.add(Conv2D(8, kernel_size=5, strides=1, padding="valid", kernel_initializer="glorot_normal",
              bias_initializer="glorot_normal"))  # Glorot normal = Xavier
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=2, strides=2))

    model.add(Conv2D(16, kernel_size=3, padding="valid",
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
    checkpoint = ModelCheckpoint(
        "best_model_w", save_best_only=True, save_weights_only=True)
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[
                        checkpoint], validation_data=(X_test, y_test))
    return history


layer_names = {1: "conv_layer1", 4: "conv_layer2",
               8: "fc_layer1", 10: "fc_layer2", 12: "fc_layer3"}


def save_layer_ws(layer: Layer, layer_index=int, w_file=IO):
    if layer_index not in layer_names.keys():
        return
    biases = np.array(layer.weights[1])
    kernels = np.array(layer.weights[0])

    print(
        f"\nfloat24_t {layer_names[layer_index]}_bias[{biases.shape[0]}] ="+"{ ", file=w_file, end="")
    for w in biases[:-1]:
        print(w, file=w_file, end=",")
    print(biases[-1], file=w_file, end="};\n")

    shape_str = "".join([f"[{sh}]" for sh in kernels.shape])
    print(
        f"\nfloat24_t {layer_names[layer_index]}_weights{shape_str} ="+"{ ", file=w_file, end="")
    k_flat = kernels.flatten()
    for w in k_flat[:-1]:
        print(w, file=w_file, end=",")
    print(k_flat[-1], file=w_file, end="};\n")


def save_ws(model: Sequential, output=str):
    with open(WEIGHT_FILE, "w") as f:

        print("//AUTO_GENERATED WITH get_params.py", file=f,
              end="\n//--------------------------------------------\n")
        print('\n#ifndef EXP_WIDTH\n#include <ap_fixed.h>\n\n#define EXP_WIDTH	16\n#define INT_WIDTH	4\n\ntypedef ap_fixed<EXP_WIDTH, INT_WIDTH> float24_t;\n\n#endif', file=f, end="")
        print("\n#ifndef __WEIGHTS_H\n#define __WEIGHTS_H\n\n",file=f)

        for idx, layer in enumerate(model.layers):
            save_layer_ws(layer, idx, f)
        
        print("\n\n#endif",file=f)

        '''with open("D:\\Documenti\\Università\\Embedded\\CNN-using-HLS\\nnet_stream\\headers\\test_img.h", "r") as img:
            txt = img.read()
            print("\n\n"+txt, file=f)'''

def get_exp_outputs(model: Sequential, tests: np.ndarray):
    pad_layer = model.get_layer(index=0)
    pad_model = Model(inputs = model.input, outputs = pad_layer.output)

    partial_models = {}
    for layer_idx in layer_names.keys():
        out_layer = model.get_layer(index=min(len(model.layers)-1,layer_idx+1))
        part_model = Model(inputs=model.input, outputs=out_layer.output)
        partial_models[layer_names[layer_idx]]=part_model
    
    for model_name, part_model in partial_models.items():
        with open(f"{REF_FOLDER}/{model_name}_ref.dat","w") as out_f:
            refs = part_model.predict(tests)
            for ref in refs:
                ref_flat = np.array(ref).flatten()
                for r in ref_flat:
                    print(r,file=out_f, end=" ")
                print("\n",file=out_f, end="")
    
    with open(f"{REF_FOLDER}/inputs_ref.dat","w") as out_f:
        pads = pad_model.predict(tests)
        for pad in pads:
            pad_flat = np.array(pad).flatten()
            for p in pad_flat:
                print(p,file=out_f, end=" ")
            print("\n",file=out_f, end="")

def main():
    global model
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.astype("float32") / 255.0
    X_train = X_train.reshape((-1, X_train.shape[1], X_train.shape[2], 1))
    X_test = X_test.astype("float32") / 255.0
    X_test = X_test.reshape((-1, X_test.shape[1], X_test.shape[2], 1))
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    model = build_model(X_train.shape[1:])
    if TRAIN:
        fit_model(model, X_train, y_train, X_test, y_test)
    model.load_weights("best_model_w")
    print(model.summary())
    #save_ws(model)
    get_exp_outputs(model,X_test[:NUM_TESTS])


if __name__ == "__main__":
    main()
