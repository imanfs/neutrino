
import h5py
import numpy as np
import matplotlib.pyplot as plt 

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import keras

from utils import train_extractor, test_extractor, image_separator, feature_extractor

# import the urllib library
import urllib.request

import enum 
class Interaction(enum.Enum):
    kNumuQE =0           # Numu CC QE interaction
    kNumuRes =1           # Numu CC Resonant interaction
    kNumuDIS = 2          # Numu CC DIS interaction
    kNumuOther = 3        # Numu CC, other than above
    kNueQE = 4            # Nue CC QE interaction
    kNueRes = 5           # Nue CC Resonant interaction
    kNueDIS = 6           # Nue CC DIS interaction
    kNueOther = 7         # Nue CC, other than above
    kNutauQE = 8          # Nutau CC QE interaction
    kNutauRes = 9         # Nutau CC Resonant interaction
    kNutauDIS =10         # Nutau CC DIS interaction
    kNutauOther =11       # Nutau CC, other than above
    kNuElectronElastic = 12# NC  Nu On E Scattering
    kNC =13                # NC interaction
    kCosmic =14           # Cosmic ray background
    kOther =15            # Something else.  Tau?  Hopefully we don't use this
    kNIntType=16          # Number of interaction types, used like a vector size


global task

urllib.request.urlretrieve('http://www.hep.ucl.ac.uk/undergrad/0056/other/projects/nova/neutrino1.h5', 'neutrino1.h5')
df=h5py.File('neutrino1.h5','r')
n_events = np.array(df['cvnmap']).shape[0]
numu, nue, nutau, DIS, QE = 0,0,0,0,0
ints = df['neutrino']['interaction']
for i in range(n_events):
  if (ints[i] <= 3):
    numu +=1 
  elif (ints[i] <= 7) & (ints[i] > 3):
    nue += 1
  elif (ints[i] <= 11) & (ints[i] > 7):
    nutau +=1
  if (ints[i] == 0) or (ints[i] == 4) or (ints[i] == 8):
    QE +=1
  elif (ints[i] == 2) or (ints[i] == 6) or (ints[i] == 10):
    DIS += 1

print("muon neutrino events make up {0:0.2f}".format(100*numu/n_events), "% of all events")
print("electrons neutrino events make up {0:0.2f}".format(100*nue/n_events), "% of all events")
print("tau neutrino events make up {0:0.2f}".format(100*nutau/n_events), "% of all events")
print("DIS events make up {0:0.2f}".format(100*DIS/n_events), "% while QE events make up {0:0.2f}".format(100*QE/n_events), "%")

########
task = 1
########

train_images, train_labels = train_extractor(39)
xz_train, yz_train  = image_separator(train_images)

ones = 0
zeros = 0
# checking that there is an equal number of both labels
for n in train_labels:
  if n == 1:
    ones += 1
  else: 
    zeros += 1
print("The number of CC events is:", ones,"and the number of non-CC events is:", zeros)

xz_input=keras.layers.Input(shape=(100,80,1))
yz_input= keras.layers.Input(shape=(100,80,1))
xz_model = feature_extractor(xz_input) # feature extractor for xz images
yz_model = feature_extractor(yz_input) # feature extractor for yz images

conv = keras.layers.concatenate([xz_model, yz_model])
model = keras.layers.Flatten()(conv)
model = keras.layers.Dense(8, activation='relu')(model)
model = keras.layers.Dense(32, activation='relu')(model)
model = keras.layers.Dropout(0.3)(model)
output = keras.layers.Dense(1, activation='sigmoid')(model)
model = keras.models.Model(inputs=[xz_input, yz_input], outputs=[output])

# summarize layers
print(model.summary())

# compiling the first model
early_stop = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", mode = "min", 
                                              patience=5, restore_best_weights=True)
keras.backend.clear_session()
model.compile(optimizer='adam',
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['binary_accuracy'])
history = model.fit([xz_train,yz_train], train_labels, epochs=70, verbose = 2,
                    validation_split=0.2,shuffle=True, batch_size=256, callbacks=early_stop)

# we plot the accuracies of the trained model on the same graph as the test data accuracy
plt.figure()
plt.plot(history.history['binary_accuracy'], label='Training accuracy')
plt.plot(history.history['val_binary_accuracy'], label = 'Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.title("Comparing the accuracies of the validation and trained data for a CC event classifier")


plt.figure()
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.title("Comparing the losses of the validation and trained data for a CC event classifier")


task = 1
test_images, test_labels, test_nuenergy, test_lepenergy,test_interactions,test_fs = test_extractor(50,1)

xz_test, yz_test = image_separator(test_images)
test_acc, test_loss = model.evaluate([xz_test, yz_test], test_labels, verbose=2)

acc = []
loss = []
for i in range(10):
  xz_input=keras.layers.Input(shape=(100,80,1))
  yz_input= keras.layers.Input(shape=(100,80,1))
  xz_model = feature_extractor(xz_input) # feature extractor for xz images
  yz_model = feature_extractor(yz_input) # feature extractor for yz images
  conv = keras.layers.concatenate([xz_model, yz_model])
  model = keras.layers.Flatten()(conv)
  model = keras.layers.Dense(8, activation='relu')(model)
  model = keras.layers.Dense(32, activation='relu')(model)
  model = keras.layers.Dropout(0.3)(model)
  output = keras.layers.Dense(1, activation='sigmoid')(model)
  model = keras.models.Model(inputs=[xz_input, yz_input], outputs=[output])
  keras.backend.clear_session()
  model.compile(optimizer='adam',
                loss=tf.keras.losses.binary_crossentropy,
                metrics=['binary_accuracy'])
  history = model.fit([xz_train,yz_train], train_labels, epochs=25,
                      validation_split=0.2,shuffle=True, verbose=0, batch_size=256, callbacks=early_stop)
  test_acc, test_loss = model.evaluate([xz_test, yz_test], test_labels, verbose=2)
  acc.append(test_loss) # for some reason the loss and accuracy were switched around, not sure why 
  loss.append(test_acc)

avg_acc, acc_error = np.mean(acc), np.std(acc)/np.sqrt(10)
avg_loss, loss_error = np.mean(loss), np.std(loss)/np.sqrt(10)

print("the average test accuracy is {0:0.2f}".format(avg_acc) , 
      "with a standard error of {0:0.2f}".format(acc_error))
print("the average test loss is {0:0.2f}".format(avg_loss) , 
      "with a standard error of {0:0.2f}".format(loss_error))


nuenergy=np.array(df['neutrino']['nuenergy'])
lepenergy=np.array(df['neutrino']['lepenergy'])
plt.figure()
plt.hist(nuenergy)
plt.xlabel("Neutrino energy values")
plt.ylabel("Number of data points")
plt.title("Neutrino energy distribution")

plt.figure()
plt.hist(lepenergy,bins = 20)
plt.xlabel("Lepton energy values")
plt.ylabel("Number of data points")
plt.title("Lepton energy distribution")


DIS,DIS_labels = [],[]
QE,QE_labels = [],[]
highnuenergy,highnuenergy_labels = [],[]
lownuenergy,lownuenergy_labels = [],[]
highlepenergy,highlepenergy_labels = [],[]
lowlepenergy,lowlepenergy_labels = [],[]

# creating mini test sets to analyse model efficiency based on metadata
for i in range(len(test_images)):
  if (test_interactions[i] == 0) or (test_interactions[i] == 4) or (test_interactions[i] == 8):
    QE.append(test_images[i]) # appending only QE events/labels
    QE_labels.append(1) if test_interactions[i] <= 3 else QE_labels.append(0)

  if (test_interactions[i] == 2) or (test_interactions[i] == 6) or (test_interactions[i] == 10):
    DIS.append(test_images[i])
    DIS_labels.append(1) if test_interactions[i] <= 3 else DIS_labels.append(0)

  if (test_nuenergy[i] <= np.max(test_nuenergy)/5):
    lownuenergy.append(test_images[i])
    lownuenergy_labels.append(1) if test_interactions[i] <= 3 else lownuenergy_labels.append(0)

  else:
    highnuenergy.append(test_images[i])
    highnuenergy_labels.append(1) if test_interactions[i] <= 3 else highnuenergy_labels.append(0) 

  if (test_lepenergy[i] >= 3.8):
    highlepenergy.append(test_images[i])
    highlepenergy_labels.append(1) if test_interactions[i] <= 3 else highlepenergy_labels.append(0)

  else:
    lowlepenergy.append(test_images[i])
    lowlepenergy_labels.append(1) if test_interactions[i] <= 3 else lowlepenergy_labels.append(0)

QE_labels,DIS_labels,highnuenergy_labels,lownuenergy_labels,highlepenergy_labels,lowlepenergy_labels = np.array(QE_labels),np.array(DIS_labels),np.array(highnuenergy_labels),np.array(lownuenergy_labels),np.array(highlepenergy_labels),np.array(lowlepenergy_labels)

### QE ###

QE_xz, QE_yz = image_separator(np.array(QE))
test_acc, test_loss = model.evaluate([QE_xz, QE_yz], QE_labels, verbose=2)

### DIS ###

DIS_xz, DIS_yz = image_separator(np.array(DIS))
test_acc, test_loss = model.evaluate([DIS_xz, DIS_yz], DIS_labels, verbose=2)

### high-nu energy ###

highnuenergy_xz, highnuenergy_yz = image_separator(np.array(highnuenergy))
test_acc, test_loss = model.evaluate([highnuenergy_xz, highnuenergy_yz], highnuenergy_labels, verbose=2)

### low nu energy ### 

lownuenergy_xz, lownuenergy_yz = image_separator(np.array(lownuenergy))
test_acc, test_loss = model.evaluate([lownuenergy_xz, lownuenergy_yz], lownuenergy_labels, verbose=2)

### high lepton energy ### 

highlepenergy_xz, highlepenergy_yz = image_separator(np.array(highlepenergy))
test_acc, test_loss = model.evaluate([highlepenergy_xz, highlepenergy_yz], highlepenergy_labels, verbose=2)

### low lepton energy

lowlepenergy_xz, lowlepenergy_yz = image_separator(np.array(lowlepenergy))
test_acc, test_loss = model.evaluate([lowlepenergy_xz, lowlepenergy_yz], lowlepenergy_labels, verbose=2)

task = 3
train_images, train_labels = train_extractor(8)
test_images, test_labels = test_extractor(11,1)
xz_train, yz_train  = image_separator(train_images)
xz_test, yz_test = image_separator(test_images)

train_labels, test_labels = np.squeeze(train_labels), np.squeeze(test_labels)
xz_input=keras.layers.Input(shape=(100,80,1))
yz_input= keras.layers.Input(shape=(100,80,1))
xz_model = feature_extractor(xz_input) # feature extractor for xz images
yz_model = feature_extractor(yz_input) # feature extractor for yz images

conv = keras.layers.concatenate([xz_model, yz_model])
model = keras.layers.Flatten()(conv)
model = keras.layers.Dense(42, activation='relu')(model)
model = keras.layers.Dense(64, activation='relu')(model)
model = keras.layers.Dropout(0.3)(model)
output = keras.layers.Dense(1, activation='sigmoid')(model)
model = keras.models.Model(inputs=[xz_input, yz_input], outputs=[output])

# summarize layers
print(model.summary())
# plot model
#plot_model(model, to_file='convolutional_neural_network.png')

# compiling the first model
early_stop = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", mode = "min", patience=5, restore_best_weights=True)
keras.backend.clear_session()
opt = tf.keras.optimizers.RMSprop()
model.compile(optimizer=opt,
              loss=tf.keras.losses.mse,
              metrics=['mse'])
history = model.fit([xz_train,yz_train], train_labels, epochs=50,
                    validation_split=0.2,shuffle=True, batch_size=256, callbacks=early_stop)

# we plot the accuracies of the trained model on the same graph as the test data accuracy
plt.figure()
plt.plot(history.history['mse'], label='Training accuracy')
plt.plot(history.history['val_mse'], label = 'Test accuracy')
plt.xlabel('Epoch')
plt.ylabel('Mean squared error')
plt.legend(loc='best')
plt.title("Comparing the accuracies and losses of the validation and trained data for a CC event classifier")


plt.figure()
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.title("Comparing the accuracies and losses of the validation and trained data for a CC event classifier")


########
task = 4
########

train_images, train_labels = train_extractor(46)
test_images, test_labels = test_extractor(60,1)
xz_train, yz_train  = image_separator(train_images)
xz_test, yz_test = image_separator(test_images)

xz_input = keras.layers.Input(shape=(100,80,1))
xz_model = feature_extractor(xz_input) # feature extractor for xz images
yz_input = keras.layers.Input(shape=(100,80,1))
yz_model = feature_extractor(yz_input) # feature extractor for yz images

conv = keras.layers.concatenate([xz_model, yz_model])
dense = keras.layers.Dense(8, activation='relu')(conv)
dense = keras.layers.Dense(32, activation='relu')(dense)
dense = keras.layers.Dropout(0.5)(dense)
output = keras.layers.Dense(1, activation='sigmoid')(dense)
model = keras.models.Model(inputs=[xz_input, yz_input], outputs=[output])

# summarize layers
print(model.summary())
# plot model
#plot_model(model, to_file='convolutional_neural_network.png')

# compiling the model
early_stop = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", mode = "min", 
                                              patience=5, restore_best_weights=True)
keras.backend.clear_session()
model.compile(optimizer='adam',
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['binary_accuracy'])
history = model.fit([xz_train,yz_train], train_labels, epochs=60,
                    validation_split=0.2,shuffle=True, batch_size=256, callbacks=early_stop)

# we plot the accuracies of the trained model on the same graph as the test data accuracy
plt.figure()
plt.plot(history.history['binary_accuracy'], label='Training accuracy')
plt.plot(history.history['val_binary_accuracy'], label = 'Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.title("Comparing the accuracies of the validation and trained data for a neutrino flavour classifier")


plt.figure()
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.title("Comparing the losses of the validation and trained data for a neutrino flavour classifier")

test_acc, test_loss = model.evaluate([xz_test, yz_test], test_labels, verbose=2)

acc = []
loss = []
for i in range(10):
  xz_input=keras.layers.Input(shape=(100,80,1))
  yz_input= keras.layers.Input(shape=(100,80,1))
  xz_model = feature_extractor(xz_input) # feature extractor for xz images
  yz_model = feature_extractor(yz_input) # feature extractor for yz images
  conv = keras.layers.concatenate([xz_model, yz_model])
  dense = keras.layers.Dense(8, activation='relu')(conv)
  dense = keras.layers.Dense(32, activation='relu')(dense)
  dense = keras.layers.Dropout(0.5)(dense)
  output = keras.layers.Dense(1, activation='sigmoid')(dense)
  model = keras.models.Model(inputs=[xz_input, yz_input], outputs=[output])
  keras.backend.clear_session()
  model.compile(optimizer='adam',
                loss=tf.keras.losses.binary_crossentropy,
                metrics=['binary_accuracy'])
  history = model.fit([xz_train,yz_train], train_labels, epochs=20,validation_split=0.2,shuffle=True, verbose=0, batch_size=256, callbacks=early_stop)
  test_acc, test_loss = model.evaluate([xz_test, yz_test], test_labels, verbose=2)
  acc.append(test_acc)
  loss.append(test_loss)



avg_acc, acc_error = np.mean(loss), np.std(loss)/np.sqrt(10)
avg_loss, loss_error = np.mean(acc), np.std(acc)/np.sqrt(10)
print("the average test accuracy is {0:0.2f}".format(avg_acc) , "with a standard error of {0:0.2f}".format(acc_error))
print("the average test loss is {0:0.2f}".format(avg_loss) , "with a standard error of {0:0.2f}".format(loss_error))


task = 5
train_images, train_labels = train_extractor(7)
test_images, test_labels = test_extractor(11,2)
xz_train, yz_train  = image_separator(train_images)
xz_test, yz_test = image_separator(test_images)

train_labels, test_labels = np.squeeze(train_labels), np.squeeze(test_labels)
xz_input=keras.layers.Input(shape=(100,80,1))
yz_input= keras.layers.Input(shape=(100,80,1))
xz_model = feature_extractor(xz_input) # feature extractor for xz images
yz_model = feature_extractor(yz_input) # feature extractor for yz images

conv = keras.layers.concatenate([xz_model, yz_model])
model = keras.layers.Flatten()(conv)
model = keras.layers.Dense(48, activation='relu')(model)
model = keras.layers.Dense(64, activation='relu')(model)
model = keras.layers.Dropout(0.3)(model)
output = keras.layers.Dense(1, activation='sigmoid')(model)
model = keras.models.Model(inputs=[xz_input, yz_input], outputs=[output])

# summarize layers
print(model.summary())
# plot model

# compiling the model
early_stop = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", mode = "min", patience=5, restore_best_weights=True)
keras.backend.clear_session()
opt = tf.keras.optimizers.RMSprop()
model.compile(optimizer=opt,
              loss=tf.keras.losses.mse,
              metrics=['mse'])
history = model.fit([xz_train,yz_train], train_labels, epochs=60,
                    validation_split=0.2,shuffle=True, batch_size=256, callbacks=early_stop)

# we plot the accuracies of the trained model on the same graph as the test data accuracy
plt.figure()
plt.plot(history.history['mse'], label='Training accuracy')
plt.plot(history.history['val_mse'], label = 'Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.title("Comparing the accuracies of the validation and trained data for regression model")


plt.figure()
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.title("Comparing the losses of the validation and trained data for regression model")


test_acc, test_loss = model.evaluate([xz_test, yz_test], test_labels, verbose=2)

task = 6
train_images, train_labels = train_extractor(15)
test_images, test_labels = test_extractor(11,2)


xz_train, yz_train  = image_separator(train_images)
xz_test, yz_test = image_separator(test_images)

# converts labels to categorical (0 to 4) from binary
# need to be careful to run this cell only once otherwise it will keep adding 
# extra dimensions
train_labels = tf.keras.utils.to_categorical(train_labels, 5)
test_labels = tf.keras.utils.to_categorical(test_labels, 5)


xz_input = keras.layers.Input(shape=(100,80,1))
xz_model = feature_extractor(xz_input) # featur#e extractor for xz images
yz_input = keras.layers.Input(shape=(100,80,1))
yz_model = feature_extractor(yz_input) # feature extractor for yz images

conv = keras.layers.concatenate([xz_model, yz_model])
dense = keras.layers.Dense(8, activation='relu')(conv)
dense = keras.layers.Dense(32, activation='relu')(dense)
dense = keras.layers.Dropout(0.5)(dense)
output = keras.layers.Dense(5, activation='softmax')(dense)
model = keras.models.Model(inputs=[xz_input, yz_input], outputs=[output])


# compiling the first model
#early_stop = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", mode = "min", 
                                            
keras.backend.clear_session()

model.compile(optimizer='adam',loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
history = model.fit([xz_train,yz_train], train_labels, epochs=100, 
                    validation_split=0.2,shuffle=True, batch_size=256, callbacks=[early_stop])

# we plot the accuracies of the trained model on the same graph as the test data accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label = 'Validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.title("Comparing the accuracies for interaction mode classifier")
plt.figure()
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.title("Comparing the losses for interaction mode classifier")
