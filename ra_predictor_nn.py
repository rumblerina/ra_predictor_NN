from pyexpat import native_encoding
import pandas as pd
import geopandas as gpd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from matplotlib.dates import DateFormatter
import numpy as np
from numpy.core.arrayprint import DatetimeFormat
from numpy.lib.histograms import _ravel_and_check_weights
from scipy import stats
from sklearn.metrics import *

df = pd.read_csv('raw_geodata.csv', header = 0, index_col=[0])
ra = pd.read_csv("xy_Ra.csv", header = 0)
ra = ra.drop(['id', 'number', 'name'], axis = 1)
ycol = ra.pop('ycoord')
xcol = ra.pop('xcoord')
ra.insert(0, 'Y', ycol)
ra.insert(0, 'X', xcol)
res = 50
df = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df['X'], df['Y']))
ra = gpd.GeoDataFrame(ra, geometry = gpd.points_from_xy(ra['X'], ra['Y']))
df = df.sjoin_nearest(ra, how = "left", max_distance=res)
df = df.drop(['geometry', 'index_right', 'X_right', 'Y_right', 'rfdk', 'rfdd', 'rfd'], axis = 1)
#df = df.dropna()
df = df.rename({'X_left':'X', 'Y_left':'Y'}, axis=1)

categoricals = ['mezolayers', 'quartlayers']
#df = df.sample(frac = 0.25)
df = pd.get_dummies(df, columns = categoricals, dummy_na=False)
#gdf = for prediction properties
gdf = df.copy()
#df = for training purposes
df = df.dropna()
df232 = df.copy()
gdf = gdf[gdf.Ra.isin(df.Ra) == False]
#df = df.astype({"X": 'float16', "Y": 'float16', 'dose': 'float16', 'mezo': 'float16', 'hght': 'float16', 'ra': 'float16', 'rfdd': 'float16'})
#training = the fraction on which learning will be done 
#testing - fraction on which how well the education was done will be checked
df_trn = df.sample(frac=0.8)
df_tst = df.drop(df_trn.index)
#features - predictors
ftr_trn = df_trn.copy()
ftr_tst = df_tst.copy()
#labels - real values
lbl_trn = ftr_trn.pop('Ra')
lbl_tst = ftr_tst.pop('Ra')

def undummify(df, prefix_sep="_"):
    cols2collapse = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df
    
    #Normalizing data and assembling the model
normalizer = preprocessing.Normalization(axis = -1)
normalizer.adapt(np.array(ftr_trn))
first = np.array(ftr_trn[:1])
Nepochs = 10000

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [t]')
    plt.legend()
    plt.grid(True)

#define the model
def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(512, activation='sigmoid'),
        layers.Dense(512, activation='sigmoid'),
        layers.Dense(256, activation='sigmoid'),
        layers.Dense(1)
    ])
    model.compile(loss='mse',
                optimizer=tf.keras.optimizers.Adam(0.001))
    return model

dnn_model = build_and_compile_model(normalizer)
#dnn_model.summary()
history = dnn_model.fit(
    ftr_trn, lbl_trn,
    batch_size = 1024,
    validation_split=0.2,
    verbose=0, epochs=Nepochs,
)

#Results
# Polynomial Regression
plot_loss(history)
plt.show()

pre_predx = dnn_model.predict(ftr_tst).flatten()
a = plt.axes(aspect='equal')
plt.scatter(lbl_tst, pre_predx)
plt.xlabel('True Values')
plt.ylabel('Predictions')
print(r2_score(lbl_tst, pre_predx))
print(mean_squared_error(lbl_tst, pre_predx))
plt.show()

error = pre_predx - lbl_tst
plt.hist(error, bins=25)
plt.xlabel('Prediction Error')
_ = plt.ylabel('Count')
plt.show()

df_pred = gdf.copy()
#df_pred = df_pred.drop(df, axis = 1)
# df_pred = df_pred.drop('rfdk', axis = 1)
#df_pred = df_pred.drop('rfd', axis = 1)
df_pred = df_pred[df_pred.Ra.isin(df.Ra) == False]
print(ftr_trn)
print(df_pred)
df_pred = df_pred.drop('Ra', axis = 1)

predix = dnn_model.predict(df_pred).flatten()
df_pred['rapred'] = predix
df_pred = df_pred[(np.abs(stats.zscore(df_pred['rapred'])) < 3)]
df_pred = undummify(df_pred, "_")
df_pred.to_csv('Ra_NN.csv', sep = ' ', header = False, index = False)
