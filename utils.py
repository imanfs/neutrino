
import h5py
import numpy as np

from sklearn.utils import shuffle

import keras

# import the urllib library
import urllib.request

global task

def train_extractor(n_files):
  '''
  Function that extracts HDF5 data into training data and labels, depending on the task
  Inputs:
  n_files: number of files to extract data from, starts at 0
  Outputs:
  maps: xy and yz images of events for training
  labels: corresponding labels for the maps
  '''
  total_n_events = 0 # initialising events
  maps = []                    # training data
  labels = []                  # training labels
  # empty arrays 
  interactions,events,nuenergy,lepenergy = [],[],[],[]            
    
  for i in range(n_files): # looping through files
    filename = str("neutrino" + str(i+1) + ".h5") # creating filename to extract data from urllib.request
    urllib.request.urlretrieve('http://www.hep.ucl.ac.uk/undergrad/0056/other/projects/nova/' + filename, filename)
    df = h5py.File(filename,'r')
    n_events = df['cvnmap'].shape[0]
    total_n_events += n_events
    
    # extracting maps and meta data 
    cvnmaps=np.array(df['cvnmap']).reshape(n_events,100,80,2) # extracting and reshaping events in each data file
    nu_e =np.array(df['neutrino']['nuenergy'])
    lep_e =np.array(df['neutrino']['lepenergy'])
    ints = np.array(df['neutrino']['interaction'])
    
    for j in range(n_events):
      # appending each event/meta variable to empty array, so we have arrays full of all events/meta data from all files
      nuenergy.append(nu_e[j])
      lepenergy.append(lep_e[j])
      interactions.append(ints[j])
      events.append(cvnmaps[j]) # creating one array full of all events from all imported files    
  
  events=np.array(events)
  zeros_maps, zeros_labels = [], []
  ones_maps, ones_labels = [], []
  QE, DIS, RES, other, NC = [], [], [], [], []
  QE_labels, DIS_labels, RES_labels, other_labels, NC_labels = [],[],[], [], []

### numu event classifier ### 
  if task == 1:
    for j in range(total_n_events):
      if interactions[j] <= 3:
        ones_labels.append(int(1))
        ones_maps.append(events[j])
      else:
        zeros_labels.append(int(0))
        zeros_maps.append(events[j])   
    ones_labels, ones_maps = ones_labels[:len(zeros_labels)], ones_maps[:len(zeros_maps)] # cutting 1s array to the same size as 0s array
    maps = zeros_maps + ones_maps
    labels = zeros_labels + ones_labels

### neutrino energy prediction algorithm ###
  elif task == 3:
    norm = max(nuenergy)
    for j in range(total_n_events):
      maps.append(events[j]) 
      labels.append(nuenergy[j]/norm)

### interaction type classifier ###
  elif task == 4:
    for j in range(total_n_events):
      if interactions[j] <= 3:
        ones_labels.append(1)
        ones_maps.append(events[j])
      elif (interactions[j] <=7) & (interactions[j] > 3):
        zeros_labels.append(0)
        zeros_maps.append(events[j])   
    ones_labels, ones_maps = ones_labels[:len(zeros_labels)], ones_maps[:len(zeros_maps)]
    maps = zeros_maps + ones_maps
    labels = zeros_labels + ones_labels

### y = lepton/neutrino energy algorithm ###
  elif task == 5:
    for j in range(total_n_events):
      if nuenergy[j] > 0:
        maps.append(events[j]) 
        labels.append((lepenergy[j]/nuenergy[j]))
  
  elif task == 6:
    for j in range(total_n_events):
      if (interactions[j] == 0) or (interactions[j] == 4) or (interactions[j] == 8):
        QE.append(events[j]) # appending only QE events/labels
        QE_labels.append(4) 
      elif (interactions[j] == 1) or (interactions[j] == 5) or (interactions[j] == 9):
        RES.append(events[j]) # appending only QE events/labels
        RES_labels.append(3) 
      elif (interactions[j] == 2) or (interactions[j] == 6) or (interactions[j] == 10):
        DIS.append(events[j]) # appending only QE events/labels
        DIS_labels.append(2)
      elif (interactions[j] == 3) or (interactions[j] == 7) or (interactions[j] == 11):
        other.append(events[j]) # appending only QE events/labels
        other_labels.append(1)
      elif (interactions[j] == 12) or (interactions[j] == 13):
        NC.append(events[j]) # appending only QE events/labels
        NC_labels.append(0)
    
    QE_labels, QE = QE_labels[:len(other_labels)], QE[:len(other)]
    RES_labels, RES = RES_labels[:len(other_labels)], RES[:len(other)]
    DIS_labels, DIS = DIS_labels[:len(other_labels)], DIS[:len(other)]
    NC_labels, NC = NC_labels[:len(other_labels)], NC[:len(other)]

    maps = QE + RES + DIS + other + NC
    labels = QE_labels + RES_labels + DIS_labels + other_labels + NC_labels
      

  maps, labels = shuffle(np.array(maps),np.array(labels)) # shuffles the data and labels simultaneously ***without disrupting the mapping***
  return maps, labels


def test_extractor(start,n_files):
  '''
  Function that extracts HDF5 data into testing data and labels, depending on the task
  Inputs:
  start: which filenumber to start from, so as to avoid re-using train data
  n_files: number of files to extract data from
  Outputs:
  maps: xy and yz images of events for testing
  labels: corresponding labels for the maps
  '''
  total_n_events = 0 # initialising events
  # empty arrays 
  maps,labels,interactions,nuenergy,lepenergy,events,fs = [],[],[],[],[],[],[]
  files = np.arange(start, start+n_files)  

  for i in files:
    filename = str("neutrino" + str(i+1) + ".h5") # creating filename to extract data from urllib.request
    urllib.request.urlretrieve('http://www.hep.ucl.ac.uk/undergrad/0056/other/projects/nova/' + filename, filename)
    df = h5py.File(filename,'r')
    n_events = df['cvnmap'].shape[0]
    total_n_events += n_events
    
    # extracting maps and meta data 
    cvnmaps=np.array(df['cvnmap']).reshape(n_events,100,80,2) # extracting and reshaping events in each data file
    finalstate =np.array(df['neutrino']['finalstate'])
    nu_e =np.array(df['neutrino']['nuenergy'])
    lep_e =np.array(df['neutrino']['lepenergy'])
    ints = np.array(df['neutrino']['interaction'])
    
    for j in range(n_events):
      # appending each event/meta variable to empty array, so we have arrays full of all events/meta data from all files
      nuenergy.append(nu_e[j])
      lepenergy.append(lep_e[j])
      interactions.append(ints[j])
      fs.append(finalstate[j])
      events.append(cvnmaps[j]) # creating one array full of all events from all imported files    
  events=np.array(events)

  if task == 1:
    for j in range(total_n_events):
      maps.append(events[j]) 
      labels.append(1) if interactions[j] <= 3 else labels.append(0)
      # all the metadata variables are extracted to investigate classifier efficiency (task 2)
    maps, labels, nuenergy, lepenergy,interactions,fs = np.array(maps), np.array(labels), np.array(nuenergy), np.array(lepenergy),np.array(interactions),np.array(fs)
    return maps, labels, nuenergy, lepenergy,interactions,fs
  
  elif task == 3:
    norm = max(nuenergy)
    for j in range(total_n_events):  
      maps.append(events[j]) # all events from here will be added to dataset
      labels.append(nuenergy[j]/norm) 
    return np.array(maps), np.array(labels)
    
  elif task == 4:
    for j in range(total_n_events):
      if interactions[j] <= 3:
        labels.append(1) # is muon neutrino event
        maps.append(events[j])
      elif (interactions[j] <=7 ) & (interactions[j] > 3):
        labels.append(0) # is electron neutrino event
        maps.append(events[j])
    maps, labels = shuffle(np.array(maps),np.array(labels))
    return maps, labels

  elif task == 5:
    for j in range(total_n_events):   
      if nuenergy[j] > 0: 
        maps.append(events[j])  # all events from here will be added to dataset
        labels.append(lepenergy[j]/nuenergy[j])
    return np.array(maps), np.array(labels)

  elif task == 6:
    for j in range(total_n_events):
      if (interactions[j] == 0) or (interactions[j] == 4) or (interactions[j] == 8):
        maps.append(events[j]) 
        labels.append(4) 
      elif (interactions[j] == 1) or (interactions[j] == 5) or (interactions[j] == 9):
        maps.append(events[j]) 
        labels.append(3) 
      elif (interactions[j] == 2) or (interactions[j] == 6) or (interactions[j] == 10):
        maps.append(events[j]) 
        labels.append(2) 
      elif (interactions[j] == 3) or (interactions[j] == 7) or (interactions[j] == 11):
        maps.append(events[j])
        labels.append(1) 
      elif (interactions[j] == 12) or (interactions[j] == 13):
        maps.append(events[j]) 
        labels.append(0) 
    maps, labels = shuffle(np.array(maps),np.array(labels))
    return maps, labels


def feature_extractor(input_img): 
  '''
  Function to create convolutional layers to extract features from events using Keras Functional API
  Inputs:
  input_img: input shape of images to be trained using keras.layers.Input
  Outputs:
  model: model that extracts features from images
  '''
  alpha = 1e-4
  if task == 1:
    model = keras.layers.Conv2D(8,(2,2), activation='relu',  padding='same', input_shape=(100,80,1), 
                  kernel_regularizer = keras.regularizers.l2(alpha),
                  bias_regularizer = keras.regularizers.l2(alpha)) (input_img)
    model = keras.layers.MaxPooling2D(pool_size=(3,3), padding='same')(model)
    model = keras.layers.Dropout(0.2)(model)
    model = keras.layers.Conv2D(32,(2,2), activation='relu', padding='same',
                   kernel_regularizer = keras.regularizers.l2(alpha),
                   bias_regularizer = keras.regularizers.l2(alpha))(model)
    model = keras.layers.MaxPooling2D(pool_size=(3,3), padding='same')(model)
    model = keras.layers.Dropout(0.2)(model) 
    model = keras.layers.Conv2D(128,(2,2), activation='relu', padding='same')(model)
    model = keras.layers.MaxPooling2D(pool_size=(3,3), padding='same')(model)
    model = keras.layers.Dropout(0.4)(model)  
    model = keras.layers.Flatten()(model)

  else:
    model = keras.layers.Conv2D(32,(2,2), activation='relu', padding='same',
                   kernel_regularizer = keras.regularizers.l2(alpha),
                   bias_regularizer = keras.regularizers.l2(alpha))(input_img)
    model = keras.layers.MaxPooling2D(pool_size=(3,3), padding='same')(model)
    model = keras.layers.Dropout(0.5)(model)
    model = keras.layers.Conv2D(16,(2,2), activation='relu', padding='same',
                   kernel_regularizer = keras.regularizers.l2(alpha),
                   bias_regularizer = keras.regularizers.l2(alpha))(model)
    model = keras.layers.MaxPooling2D(pool_size=(3,3), padding='same')(model)
    model = keras.layers.Dropout(0.3)(model)
    model = keras.layers.Conv2D(2,(2,2), activation='relu', padding='same')(model)
    model = keras.layers.MaxPooling2D(pool_size=(3,3), padding='same')(model)
    model = keras.layers.Flatten()(model)
  
  return model

def image_separator(img):
  '''
  Function to separate images into xz and yz components
  Inputs: image that we want to separate
  Outputs:
  xz: xz view of interaction
  yz: yz view of interaction
  '''
  xz = img[:,:,:,:1] 
  yz = img[:,:,:,1:] 

  return xz, yz
