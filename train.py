import caer 
import os 
from tensorflow.keras.utils import to_categorical #type: ignore
from tensorflow.keras.callbacks import LearningRateScheduler #type: ignore
from tensorflow.keras.models import load_model #type: ignore
import gc
import canaro
import pickle

# var
IMG_SIZE = (100,100)
channels = 1 
char_path = r'simpson-recog/train-data/simpsons_dataset'
model_path = r'simpson-recog/simpsons_model.keras'
training_df_path = r'simpson-recog/training_df.pkl'

# dict
char_dict = {}
for char in os.listdir(char_path):
    char_dict[char] = len(os.listdir(os.path.join(char_path, char)))

# sort
char_dict = caer.sort_dict(char_dict, descending=True)

characters = []
count = 0
for i in char_dict:
    characters.append(i[0])
    count += 1
    if count >= 10:
        break

# Create training set => check if the training set is here already -> open it instead of recreating it each time
if os.path.exists(training_df_path):
    with open(training_df_path, 'rb') as file:
        training_df = pickle.load(file)
    print("Successfully loaded existing training DF")
else: 
    training_df = caer.preprocess_from_dir(char_path, characters, channels=channels, IMG_SIZE=IMG_SIZE, isShuffle=True)
    with open(training_df_path, 'wb') as file:
        pickle.dump(training_df, file)
    print("Successfully written training DF to file")


featureSet, labels = caer.sep_train(training_df, IMG_SIZE=IMG_SIZE)

# normalizing the featureSet => (0,1)
featureSet = caer.normalize(featureSet)
labels = to_categorical(labels, len(characters))
x_train, x_val, y_train, y_val = caer.train_val_split(featureSet, labels, val_ratio=.2)

# delete 
del training_df
del featureSet
del labels
gc.collect()

# image data generator 
BATCH_SIZE = 32
EPOCHS = 10
datagen = canaro.generators.imageDataGenerator()
train_gen = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

# Creating the model 
if os.path.exists(model_path):
    model = load_model(model_path)
    print("Loaded model from model path")
else:
    model = canaro.models.createSimpsonsModel(IMG_SIZE=IMG_SIZE, channels=channels, output_dim=len(characters), loss='binary_crossentropy', decay=1e-7, learning_rate=0.001, momentum=0.7, nesterov=True)
    callbacks_list = [LearningRateScheduler(canaro.lr_schedule)]
    print(model.summary())
    training = model.fit(train_gen, 
                        steps_per_epoch=len(x_train)//BATCH_SIZE, 
                        epochs=EPOCHS,
                        validation_data=(x_val, y_val),
                        validation_steps=len(y_val)//BATCH_SIZE,
                        callbacks=callbacks_list)
    model.save(model_path)
    print('model saved')