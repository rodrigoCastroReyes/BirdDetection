import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import tifffile as tiff
import itertools
import math
from math import sqrt
from scipy.ndimage import filters
from skimage.color import rgb2gray
from os.path import isfile, join
from skimage.feature import hog
from skimage import data, color, exposure
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from skimage.filters.rank import maximum
from skimage import morphology
from skimage import measure

from scipy.stats import zscore
import seaborn as sns
from sklearn.model_selection import KFold,StratifiedKFold,cross_val_score
from sklearn.metrics import jaccard_similarity_score
from sklearn.externals import joblib
from scipy import ndimage
from skimage import segmentation
from skimage.filters import gaussian
from os.path import isfile, join,exists
#script pata generar modelo PCA y SVM, recibe como entrada archivos .csv con los vectores de caracteristicas ver script extract_features.py

labels = {
	"ArbolA": "tree",
	"ArbolB": "tree",
	"Sombra": "shadow",
	"juvenil": "bird",
	"Madera": "tree",
	"None": "tree",
	"hembra": "bird_h",
	"macho": "bird_m",
	"shadow" : 0,
	"bird" : 1,
	"tree": 2
}

n_features = 10
begin_feature = 0
end_feature = 2*n_features + 4

dir_filename = sys.argv[1]
outputs_dir = sys.argv[2]

labels_filename = os.path.join(dir_filename,"labels")
dfs = []

for dir_input in os.listdir(labels_filename):#iterar carpeta data/labels
	df = pd.read_csv(os.path.join(labels_filename,dir_input))
	dfs.append(df)

training_set = pd.concat(dfs)
training_set['label'] = training_set['label'].apply(lambda label: labels[label])#convertir labels en numeros


df_shadows = training_set.loc[(training_set['label'] == 'shadow')]
df_tree = training_set.loc[(training_set['label'] == 'tree')]
df_birds = training_set.loc[(training_set['label'] == 'bird')]
training_set = pd.concat([df_shadows,df_tree,df_birds])

pca = PCA(n_components=0.98)
data_r = pca.fit_transform(training_set.iloc[:,begin_feature:end_feature].apply(zscore))
rows,n_components = data_r.shape
print "number of components ", n_components

for n in range(0,n_components):
	training_set['pc' + str(n)] = data_r[:,n]

sns.lmplot(x = 'pc0', y = 'pc1', data = training_set,hue="label",palette = "husl",fit_reg=False)
plt.show()

training_set_data = training_set.iloc[:,end_feature+1:]
training_set_labels = training_set['label']
training_set['label'] = training_set['label'].apply(lambda label: labels[label])

C = 1.0 #SVM regularization parameter
kf = StratifiedKFold(n_splits=5)
clfs = []
scores = []
predicted = []
test = []
for train_index, test_index in kf.split(training_set_data,training_set_labels):
	X_train, X_test = training_set_data.iloc[train_index], training_set_data.iloc[test_index]
	Y_train, Y_test = training_set_labels.iloc[train_index], training_set_labels.iloc[test_index]
	#clf = svm.LinearSVC(probability=True)
	#clf = svm.SVC(kernel='rbf', C=C,gamma=0.5,probability=True)
	clf = svm.SVC(kernel='linear',C=C,probability=True)
	clf.fit(X_train,Y_train)
	Y_predicted = clf.predict(X_test)
	score = jaccard_similarity_score(Y_test,Y_predicted)
	print score,len(Y_test)
	test.append(test_index)
	predicted.append(Y_predicted)
	clfs.append(clf)
	scores.append(score)

best_accuracy = np.argsort(scores)[::-1][0]
clf = clfs[best_accuracy]
test_index = test[best_accuracy]
best_prediction = predicted[best_accuracy]

"""
joblib.dump(pca,join(outputs_dir,'pca.pkl'))
joblib.dump(clf,join(outputs_dir,'svm.pkl'))
"""