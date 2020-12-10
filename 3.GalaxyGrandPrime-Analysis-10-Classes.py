#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# ### Functions

# In[2]:


def getData(cfileName):
    """
    Given a name of a *.cfile, this function extracts the interleaved
    Inphase-Quadrature data samples and convert it into a numpy array of complex
    data elements. *.cfile format has interleaved I and Q samples where each sample
    is a float32 type. GNURadio Companion (GRC) scripts output data into a file
    though a file sink block in this format.
    Read more in SDR data types: https://github.com/miek/inspectrum
    """
    # Read the *.cfile which has each element in float32 format.
    data = np.fromfile(cfileName, dtype="float32")
    # Take each consecutive interleaved I sample and Q sample to create a single complex element.
    data = data[0::2] + 1j*data[1::2]
    #print("data type=", type(data))
    # Return the complex numpy array.
    return data


# In[ ]:





# In[ ]:





# In[3]:


'''
data = getData("./smartphone-EM-dataset-2020-11-23/galaxy-grand-prime/audio-recording.cfile")
fig = plt.figure()
plt.psd(data, NFFT=1024, Fs=20e6)
#plt.show()
plt.savefig('GalaxyGrandPrime-audio-recording-psd.pdf', format='pdf', bbox_inches='tight')
del data


# In[4]:


data = getData("./smartphone-EM-dataset-2020-11-23/galaxy-grand-prime/camera-photo.cfile")
fig = plt.figure()
plt.psd(data, NFFT=1024, Fs=20e6)
#plt.show()
plt.savefig('GalaxyGrandPrime-camera-photo-psd.pdf', format='pdf', bbox_inches='tight')
del data


# In[5]:


data = getData("./smartphone-EM-dataset-2020-11-23/galaxy-grand-prime/camera-video.cfile")
fig = plt.figure()
plt.psd(data, NFFT=1024, Fs=20e6)
#plt.show()
plt.savefig('GalaxyGrandPrime-web-video-psd.pdf', format='pdf', bbox_inches='tight')
del data


# In[6]:


data = getData("./smartphone-EM-dataset-2020-11-23/galaxy-grand-prime/email-app.cfile")
fig = plt.figure()
plt.psd(data, NFFT=1024, Fs=20e6)
#plt.show()
plt.savefig('GalaxyGrandPrime-email-app-psd.pdf', format='pdf', bbox_inches='tight')
del data


# In[8]:


data = getData("./smartphone-EM-dataset-2020-11-23/galaxy-grand-prime/gallary-app.cfile")
fig = plt.figure()
plt.psd(data, NFFT=1024, Fs=20e6)
#plt.show()
plt.savefig('GalaxyGrandPrime-gallary-app-psd.pdf', format='pdf', bbox_inches='tight')
del data


# In[9]:


data = getData("./smartphone-EM-dataset-2020-11-23/galaxy-grand-prime/home-screen.cfile")
fig = plt.figure()
plt.psd(data, NFFT=1024, Fs=20e6)
#plt.show()
plt.savefig('GalaxyGrandPrime-home-screen-psd.pdf', format='pdf', bbox_inches='tight')
del data


# In[10]:


data = getData("./smartphone-EM-dataset-2020-11-23/galaxy-grand-prime/idle.cfile")
fig = plt.figure()
plt.psd(data, NFFT=1024, Fs=20e6)
#plt.show()
plt.savefig('GalaxyGrandPrime-idle-psd.pdf', format='pdf', bbox_inches='tight')
del data


# In[11]:


data = getData("./smartphone-EM-dataset-2020-11-23/galaxy-grand-prime/phone-app.cfile")
fig = plt.figure()
plt.psd(data, NFFT=1024, Fs=20e6)
#plt.show()
plt.savefig('GalaxyGrandPrime-phone-app-psd.pdf', format='pdf', bbox_inches='tight')
del data


# In[12]:


data = getData("./smartphone-EM-dataset-2020-11-23/galaxy-grand-prime/sms-app.cfile")
fig = plt.figure()
plt.psd(data, NFFT=1024, Fs=20e6)
#plt.show()
plt.savefig('GalaxyGrandPrime-sms-app-psd.pdf', format='pdf', bbox_inches='tight')
del data


# In[13]:


data = getData("./smartphone-EM-dataset-2020-11-23/galaxy-grand-prime/web-browser-app.cfile")
fig = plt.figure()
plt.psd(data, NFFT=1024, Fs=20e6)
#plt.show()
plt.savefig('GalaxyGrandPrime-web-browser-psd.pdf', format='pdf', bbox_inches='tight')
del data
'''

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Loading an EM Trace and Plotting

# #### Settings:

# In[3]:


# number of samples per class
num_samp_per_class = 10000

# FFT size for the STFT operation (which is same as the feature vector size)
fft_size = feature_vector_size = 2048 #1024

# number of overlapping samples for the STFT operation
fft_overlap = 256


# In[4]:


labels = ["audio-recording", "camera-photo", "camera-video", "email-app", "gallary-app", "home-screen", "idle", "phone-app", "sms-app", "web-browser-app"]

#labels = ["camera-video", "home-screen", "idle", "web-browser-app"]

#labels = ["audio-recording", "camera-video", "idle", "web-browser-app"]


# #### Preparing the Data of Class 0

# In[5]:


class_label = 0
data = getData("./smartphone-EM-dataset-2020-11-23/galaxy-grand-prime/audio-recording.cfile")
f, t, Zxx = signal.stft(data, fs=20e6, nperseg=fft_size, noverlap=fft_overlap)
print(len(f))
print(len(t))
print(Zxx.shape)
del data
Zxx = Zxx.transpose()

Zxx = abs(Zxx)
# scaling the features (only real part of the data can be used)
scaler = MinMaxScaler()
scaler.fit(Zxx)
print(scaler.data_max_)
Zxx = scaler.transform(Zxx)

X = Zxx[:num_samp_per_class]
y = np.full(num_samp_per_class, class_label)
del Zxx
print(X.shape)
print(y.shape)


# #### Preparing the Data of Class 1

# In[6]:


class_label = 1
data = getData("./smartphone-EM-dataset-2020-11-23/galaxy-grand-prime/camera-photo.cfile")
f, t, Zxx = signal.stft(data, fs=20e6, nperseg=fft_size, noverlap=fft_overlap)
print(len(f))
print(len(t))
print(Zxx.shape)
del data
Zxx = Zxx.transpose()

Zxx = abs(Zxx)
# scaling the features (only real part of the data can be used)
scaler = MinMaxScaler()
scaler.fit(Zxx)
print(scaler.data_max_)
Zxx = scaler.transform(Zxx)

X = np.concatenate((X, Zxx[:num_samp_per_class]), axis=0) 
y = np.concatenate((y, np.full(num_samp_per_class, class_label)), axis=0)
#X = Zxx[:num_samp_per_class]
#y = np.full(num_samp_per_class, class_label)
del Zxx
print(X.shape)
print(y.shape)


# #### Preparing the Data of Class 2

# In[7]:


class_label = 2
data = getData("./smartphone-EM-dataset-2020-11-23/galaxy-grand-prime/camera-video.cfile")
f, t, Zxx = signal.stft(data, fs=20e6, nperseg=fft_size, noverlap=fft_overlap)
print(len(f))
print(len(t))
print(Zxx.shape)
del data
Zxx = Zxx.transpose()

Zxx = abs(Zxx)
# scaling the features (only real part of the data can be used)
scaler = MinMaxScaler()
scaler.fit(Zxx)
print(scaler.data_max_)
Zxx = scaler.transform(Zxx)

X = np.concatenate((X, Zxx[:num_samp_per_class]), axis=0) 
y = np.concatenate((y, np.full(num_samp_per_class, class_label)), axis=0)
#X = Zxx[:num_samp_per_class]
#y = np.full(num_samp_per_class, class_label)
del Zxx
print(X.shape)
print(y.shape)


# #### Preparing the Data of Class 3

# In[8]:


class_label = 3
data = getData("./smartphone-EM-dataset-2020-11-23/galaxy-grand-prime/email-app.cfile")
f, t, Zxx = signal.stft(data, fs=20e6, nperseg=fft_size, noverlap=fft_overlap)
print(len(f))
print(len(t))
print(Zxx.shape)
del data
Zxx = Zxx.transpose()

Zxx = abs(Zxx)
# scaling the features (only real part of the data can be used)
scaler = MinMaxScaler()
scaler.fit(Zxx)
print(scaler.data_max_)
Zxx = scaler.transform(Zxx)

X = np.concatenate((X, Zxx[:num_samp_per_class]), axis=0) 
y = np.concatenate((y, np.full(num_samp_per_class, class_label)), axis=0)
#X = Zxx[:num_samp_per_class]
#y = np.full(num_samp_per_class, class_label)
del Zxx
print(X.shape)
print(y.shape)


# In[ ]:





# In[ ]:





# In[ ]:


# #### Preparing the Data of Class 4

# In[ ]:


class_label = 4
data = getData("./smartphone-EM-dataset-2020-11-23/galaxy-grand-prime/gallary-app.cfile")
f, t, Zxx = signal.stft(data, fs=20e6, nperseg=fft_size, noverlap=fft_overlap)
print(len(f))
print(len(t))
print(Zxx.shape)
del data
Zxx = Zxx.transpose()

Zxx = abs(Zxx)
# scaling the features (only real part of the data can be used)
scaler = MinMaxScaler()
scaler.fit(Zxx)
print(scaler.data_max_)
Zxx = scaler.transform(Zxx)

X = np.concatenate((X, Zxx[:num_samp_per_class]), axis=0) 
y = np.concatenate((y, np.full(num_samp_per_class, class_label)), axis=0)
#X = Zxx[:num_samp_per_class]
#y = np.full(num_samp_per_class, class_label)
del Zxx
print(X.shape)
print(y.shape)


# #### Preparing the Data of Class 5

# In[ ]:


class_label = 5
data = getData("./smartphone-EM-dataset-2020-11-23/galaxy-grand-prime/home-screen.cfile")
f, t, Zxx = signal.stft(data, fs=20e6, nperseg=fft_size, noverlap=fft_overlap)
print(len(f))
print(len(t))
print(Zxx.shape)
del data
Zxx = Zxx.transpose()

Zxx = abs(Zxx)
# scaling the features (only real part of the data can be used)
scaler = MinMaxScaler()
scaler.fit(Zxx)
print(scaler.data_max_)
Zxx = scaler.transform(Zxx)

X = np.concatenate((X, Zxx[:num_samp_per_class]), axis=0) 
y = np.concatenate((y, np.full(num_samp_per_class, class_label)), axis=0)
#X = Zxx[:num_samp_per_class]
#y = np.full(num_samp_per_class, class_label)
del Zxx
print(X.shape)
print(y.shape)


# #### Preparing the Data of Class 6

# In[ ]:


class_label = 6
data = getData("./smartphone-EM-dataset-2020-11-23/galaxy-grand-prime/idle.cfile")
f, t, Zxx = signal.stft(data, fs=20e6, nperseg=fft_size, noverlap=fft_overlap)
print(len(f))
print(len(t))
print(Zxx.shape)
del data
Zxx = Zxx.transpose()

Zxx = abs(Zxx)
# scaling the features (only real part of the data can be used)
scaler = MinMaxScaler()
scaler.fit(Zxx)
print(scaler.data_max_)
Zxx = scaler.transform(Zxx)

X = np.concatenate((X, Zxx[:num_samp_per_class]), axis=0) 
y = np.concatenate((y, np.full(num_samp_per_class, class_label)), axis=0)
#X = Zxx[:num_samp_per_class]
#y = np.full(num_samp_per_class, class_label)
del Zxx
print(X.shape)
print(y.shape)


# #### Preparing the Data of Class 7

# In[ ]:


class_label = 7
data = getData("./smartphone-EM-dataset-2020-11-23/galaxy-grand-prime/phone-app.cfile")
f, t, Zxx = signal.stft(data, fs=20e6, nperseg=fft_size, noverlap=fft_overlap)
print(len(f))
print(len(t))
print(Zxx.shape)
del data
Zxx = Zxx.transpose()

Zxx = abs(Zxx)
# scaling the features (only real part of the data can be used)
scaler = MinMaxScaler()
scaler.fit(Zxx)
print(scaler.data_max_)
Zxx = scaler.transform(Zxx)

X = np.concatenate((X, Zxx[:num_samp_per_class]), axis=0) 
y = np.concatenate((y, np.full(num_samp_per_class, class_label)), axis=0)
#X = Zxx[:num_samp_per_class]
#y = np.full(num_samp_per_class, class_label)
del Zxx
print(X.shape)
print(y.shape)


# #### Preparing the Data of Class 8

# In[ ]:


class_label = 8
data = getData("./smartphone-EM-dataset-2020-11-23/galaxy-grand-prime/sms-app.cfile")
f, t, Zxx = signal.stft(data, fs=20e6, nperseg=fft_size, noverlap=fft_overlap)
print(len(f))
print(len(t))
print(Zxx.shape)
del data
Zxx = Zxx.transpose()

Zxx = abs(Zxx)
# scaling the features (only real part of the data can be used)
scaler = MinMaxScaler()
scaler.fit(Zxx)
print(scaler.data_max_)
Zxx = scaler.transform(Zxx)

X = np.concatenate((X, Zxx[:num_samp_per_class]), axis=0) 
y = np.concatenate((y, np.full(num_samp_per_class, class_label)), axis=0)
#X = Zxx[:num_samp_per_class]
#y = np.full(num_samp_per_class, class_label)
del Zxx
print(X.shape)
print(y.shape)


# #### Preparing the Data of Class 9

# In[ ]:


class_label = 9
data = getData("./smartphone-EM-dataset-2020-11-23/galaxy-grand-prime/web-browser-app.cfile")
f, t, Zxx = signal.stft(data, fs=20e6, nperseg=fft_size, noverlap=fft_overlap)
print(len(f))
print(len(t))
print(Zxx.shape)
del data
Zxx = Zxx.transpose()

Zxx = abs(Zxx)
# scaling the features (only real part of the data can be used)
scaler = MinMaxScaler()
scaler.fit(Zxx)
print(scaler.data_max_)
Zxx = scaler.transform(Zxx)

X = np.concatenate((X, Zxx[:num_samp_per_class]), axis=0) 
y = np.concatenate((y, np.full(num_samp_per_class, class_label)), axis=0)
#X = Zxx[:num_samp_per_class]
#y = np.full(num_samp_per_class, class_label)
del Zxx
print(X.shape)
print(y.shape)


# ### Building the Model

# In[9]:

'''
X = abs(X)
# scaling the features (only real part of the data can be used)
scaler = MinMaxScaler()
scaler.fit(X)
print(scaler.data_max_)
X = scaler.transform(X)
'''

# In[ ]:





# In[ ]:





# In[ ]:





# In[10]:


# Split data to train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)


# In[11]:


model = keras.models.Sequential()
model.add(keras.layers.Input(shape=(feature_vector_size,)))
#model.add(keras.layers.Input(shape=(100,)))

model.add(keras.layers.Dense(1400, activation="relu"))

model.add(keras.layers.Dense(800, activation="relu"))
model.add(keras.layers.Dense(500, activation="relu"))
model.add(keras.layers.Dense(200, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
#model.add(keras.layers.Dense(50, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
#model.add(keras.layers.Dense(4, activation="softmax"))
model.summary()


# In[12]:


opt = keras.optimizers.SGD(learning_rate=0.001)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
#model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])


# In[13]:


#history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))
#history = model.fit(X_train, y_train, epochs=30, validation_split=0.1)

checkpoint_cb = keras.callbacks.ModelCheckpoint("./3.GalaxyGrandPrime-Analysis-10-Classes.h5", save_best_only=True)
history = model.fit(X_train, y_train, epochs=10, validation_split=0.1, callbacks=[checkpoint_cb])


# ### Plotting

# In[14]:


pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
#plt.show()
plt.savefig('learning-curve.pdf', format='pdf', bbox_inches='tight')

# ### Testing the Model

# In[15]:

model = keras.models.load_model("./3.GalaxyGrandPrime-Analysis-10-Classes.h5")


# In[16]:


results = model.evaluate(X_test, y_test)
print("test loss, test acc:", results)


# In[17]:


#y_pred = model.predict(X_test)
y_pred = model.predict_classes(X_test)


# In[18]:


print(y_pred)


# In[19]:


print(y_pred[0])


# In[20]:


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# ### Saving the Model

# In[ ]:


#model.save("./4.Nokia-4.2-Analysis-4-Classes.h5")


# In[ ]:




