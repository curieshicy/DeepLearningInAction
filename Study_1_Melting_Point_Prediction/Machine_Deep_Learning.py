'''
The file contains code for using Karthikeyan's melting point data to predict new melting points.
The dataset has 205 columns (last 203 columns containing nummerical data); 4450 rows (49 of them miss data).
Thus the total usable data points are 203*(4450 - 49) = 893403
Various machine learning algorithms are tested via shallow (Scikit_Learn) and deep learning (Keras). 

The code is written by Chenyang Shi at AbbVie. Send him an email at chenyang.shi@abbvie.com for questions/comments.

'''
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

"""
Step 1: Data cleaning
The excel dataset contains 4450 rows of data, but 49 rows are miss data; the code snippet discards these 49 rows.
Use only 4401 rows instead.
"""
##Load the metadata, the format is in Excel, thus use read_excel instead
data = pd.read_excel("Karthikeyan_MP.xls")
mp = data["MTP"]

##Drop three columns of data, "case", "SMILES" and "MTP" columns
features = data.drop(["Case", "SMILES", "MTP"], axis = 1)

##Find out index of rows with missing data
index = []
for idx, row in features.iterrows():
    if any(row.isnull()) == True:
        index.append(idx)
print (index)
print (len(index))

##Now store the features and melting points labels into new variables.
featuresdropped = features.drop(features.index[index]) ## 4401 rows x 202 columns
mpdropped = mp.drop(mp.index[index])                   ## 4401 rows x  1  column

print (featuresdropped.shape)
print (mpdropped.shape)


"""
Step 2 Perform a deep learning using the procedure laid out by Francois Chollet. Using all features

"""
# Randomly split data to contain 80% of training data, 20% of testing data, random state is 0
X_train, X_test, y_train, y_test = train_test_split(featuresdropped, mpdropped, test_size = 0.2, random_state = 0)
ss = StandardScaler()

X_train_ss = ss.fit_transform(X_train)
X_test_ss = ss.transform(X_test)

print (X_train_ss.shape) # (3520, 202) 
print (X_test_ss.shape)  # (881, 202)
##print (X_test_ss)


def sequential_model_fit(num_n1, num_n2, num_n3, num_n4, num_n5, num_n6, dropout, epochs):
    model = Sequential()
    model.add(Dense(num_n1, input_dim=202, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(num_n2, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(num_n3, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(num_n4, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(num_n5, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(num_n6, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', metrics = ["mae"], optimizer='rmsprop')

    # Train the model
    history = model.fit(
        X_train_ss,
        y_train,
        epochs=epochs,
        shuffle = False,
        verbose=2,
        validation_data = (X_test_ss, y_test)
        )
    list_params = [num_n1, num_n2, num_n3, num_n4, num_n5, num_n6, dropout, epochs]
    return history, list_params

def plot_train_val_loss_mae(his, list_p):

    a,b,c,d,e,f,g,h = list_p[0],list_p[1],list_p[2],list_p[3],list_p[4],list_p[5], list_p[6], list_p[7]
    loss = his.history["loss"]
    val_loss = his.history["val_loss"]

    mae = his.history["mean_absolute_error"]
    val_mae = his.history["val_mean_absolute_error"]
    epochs = range(1, len(loss) + 1)
    
    fig = plt.figure(figsize = (10, 6),dpi = 100)
    
    fig.add_subplot(211)
    plt.plot(epochs, loss, "bo", label = "Training loss")
    plt.plot(epochs, val_loss, "b",
             label = "Validation loss; neurons={0:3d}, {1:3d}, {2:3d}, {3:3d}, {4:3d}, {5:3d};dropout = {6:.3f};epochs = {7:3d}".format(a,b,c,d,e,f,g,h))
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xticks(fontsize = 11)
    plt.yticks(fontsize = 11)
    plt.legend(loc=0)


    fig.add_subplot(212)
    plt.plot(epochs, mae, "ro", label = "Training mean absolute error")
    plt.plot(epochs, val_mae, "r", label = "Validation mean absolute error")
    #plt.title("Training and validation MAE")
    plt.xlabel("Epochs")
    plt.ylabel("MAE")
    plt.xticks(fontsize = 11)
    plt.yticks(fontsize = 11)
    plt.legend(loc=0)
 
    fig.tight_layout()

    #plt.show()
    plt.savefig("neurons_{0:3d}_{1:3d}_{2:3d}_{3:3d}_{4:3d}_{5:3d}_dropout_{6:.3f}_epochs_{7:3d}.png".format(a,b,c,d,e,f,g,h))

##test on four layers of neurons
history = sequential_model_fit(100, 100, 100, 100, 100, 100, 0.0, 100) # neurons # 1, 2, 3, dropout, epochs
plot_train_val_loss_mae(history[0], history[1])
history = sequential_model_fit(200, 200, 200, 200, 200, 200, 0.0, 100) # neurons # 1, 2, 3, dropout, epochs
plot_train_val_loss_mae(history[0], history[1])
history = sequential_model_fit(300, 300, 300, 300, 300, 300, 0.0, 100) # neurons # 1, 2, 3, dropout, epochs
plot_train_val_loss_mae(history[0], history[1])
history = sequential_model_fit(400, 400, 400, 400, 400, 400, 0.0, 100) # neurons # 1, 2, 3, dropout, epochs
plot_train_val_loss_mae(history[0], history[1])


history = sequential_model_fit(200, 200, 200, 200, 0.0, 100) # neurons # 1, 2, 3, dropout, epochs
plot_train_val_loss_mae(history[0], history[1])
history = sequential_model_fit(200, 200, 200, 200, 0.0, 100) # neurons # 1, 2, 3, dropout, epochs
plot_train_val_loss_mae(history[0], history[1])
history = sequential_model_fit(200, 200, 200, 200, 0.0, 100) # neurons # 1, 2, 3, dropout, epochs
plot_train_val_loss_mae(history[0], history[1])
history = sequential_model_fit(200, 200, 200, 200, 0.0, 100) # neurons # 1, 2, 3, dropout, epochs
plot_train_val_loss_mae(history[0], history[1])
history = sequential_model_fit(200, 200, 200, 200, 0.0, 100) # neurons # 1, 2, 3, dropout, epochs
plot_train_val_loss_mae(history[0], history[1])
    


"""
Step 2
Use StandardScaler to preprocess data. Both X_train and X_test are tranformed while y_train, y_test are not.
Standard Machine Learning algorithms such as
DecisionTreeRegressor, LinearRegression, KNeighborsRegresson and SVRregression algorithms are tested.
"""

X_train, X_test, y_train, y_test = train_test_split(featuresdropped, mpdropped, test_size = 0.2, random_state = 0)
ss = StandardScaler()
X_train_ss = ss.fit_transform(X_train)
X_test_ss = ss.transform(X_test)

tree = DecisionTreeRegressor().fit(X_train_ss, y_train)
linear_reg = LinearRegression().fit(X_train_ss, y_train)
kneigh_reg = KNeighborsRegressor(5, "distance").fit(X_train_ss, y_train)
svr_reg = SVR().fit(X_train_ss, y_train)

pred_tree = tree.predict(X_test_ss)
pred_linear = linear_reg.predict(X_test_ss)
pred_kneigh = kneigh_reg.predict(X_test_ss)
pred_svr = svr_reg.predict(X_test_ss)


"""
Step 3

Now use Keras to peform a fit. 

"""
# Define the model
model = Sequential()
model.add(Dense(30, input_dim=202, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='rmsprop')

# Train the model
model.fit(
    X_train_ss,
    y_train,
    epochs=100,
    shuffle = False,
    verbose=2
)

DNN_pred = model.predict(X_test_ss)
print (mean_absolute_error(y_test, DNN_pred))
#model.save("DNN_50_100_50_epochs_50.h5")
#print ("Model saved to disk")

print ("###########################################This is RMSE############################################")
print ("Decision tree prediction accuracy {:.3f}".format(np.sqrt(mean_squared_error(y_test, pred_tree))))
print ("Linear regression prediction accuracy {:.3f}".format(np.sqrt(mean_squared_error(y_test, pred_linear))))
print ("K nearest neighbor regression prediction accuracy {:.3f}".format(np.sqrt(mean_squared_error(y_test, pred_kneigh))))
print ("Support vector machine prediction accuracy {:.3f}".format(np.sqrt(mean_squared_error(y_test, pred_svr))))
print ("Deep Neural Network with 3 layers, 50, 100, 50 units, prediction accuracy {:.3f}".format(np.sqrt(mean_squared_error(y_test, DNN_pred))))

print ("############################################This is MAE############################################")
print ("Decision tree prediction accuracy {:.3f}".format(mean_absolute_error(y_test, pred_tree)))
print ("Linear regression prediction accuracy {:.3f}".format(mean_absolute_error(y_test, pred_linear)))
print ("K nearest neighbor regression prediction accuracy {:.3f}".format(mean_absolute_error(y_test, pred_kneigh)))
print ("Support vector machine prediction accuracy {:.3f}".format(mean_absolute_error(y_test, pred_svr)))
print ("Deep Neural Network with 3 layers, 50, 100, 50 units, prediction accuracy {:.3f}".format(mean_absolute_error(y_test, DNN_pred)))

