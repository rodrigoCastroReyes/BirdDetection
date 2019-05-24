import numpy as np
import pandas as pd
import sys
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

def get_laplacian(img_hr_rgb,n_features):
	image_gray = rgb2gray(img_hr_rgb)
	scales = np.linspace(2.5,7.0,num=n_features)
	laplacians = []

	for scale in scales:
		img_laplacian = -1.0*filters.gaussian_laplace(image_gray,scale)
		img_laplacian = 1.0*img_laplacian/img_laplacian.max()
		laplacians.append(img_laplacian)
		#r = scale*math.sqrt(2)
		#print img_laplacian.max()
		#img_laplacian_max = maximum(img_laplacian, disk(r))
	laplacians = np.array(laplacians)
	
	return laplacians


def extract_data(img_rgb,laplacians,df,dir_input):
	image_gray = rgb2gray(img_rgb)

	img_red = img_rgb[:,:,0]
	img_red = gaussian(img_red, 3)
	s = 1.5/sqrt(2)
	img_log = ndimage.gaussian_laplace(img_red, s) * s ** 2
	img_log = np.float32(img_log)

	t_scales,row,col = laplacians.shape

	training_set = np.array([])

	#fig, ax = plt.subplots(nrows=1, ncols=1)
	#ax.set_title("RGB Image")
	#ax.imshow(image_gray,interpolation='nearest')
	for index,rows in df.iterrows():
		x = rows['x']#col
		y = rows['y']#fila
		w = rows['w']
		h = rows['h']
		label = rows['label']
		
		mask = np.zeros(image_gray.shape)

		if label == 'juvenil':
			r = w/2 if w < h else h/2
			s = np.linspace(0, 2*np.pi, 400)
			px = x + r + r*np.cos(s)
			py = y + r + r*np.sin(s)
			init = np.array([px, py]).T
			snake = segmentation.active_contour(img_log,init,alpha=0.5, beta=0.5,w_line=-0.4)
			px,py = snake[:,0].astype(int),snake[:,1].astype(int)
			#ax.plot(snake[:, 0], snake[:, 1], '-r', lw=3)
			#plt.show()
		else:
			px = [ i for i in range(x,x+w)]
			py = [ i for i in range(y,y+h)]
		
		"""
		py = [ i for i in range(x,x+w)]
		px = [ i for i in range(y,y+h)]
		"""
		positions = []
		for x,y in itertools.product(px,py):
			positions.append([x,y])
		positions = np.array(positions)
		px,py = positions[:,0],positions[:,1]

		features = np.array([])
		region = laplacians[:,py,px]

		for scale in range(t_scales):
			mean = np.mean(region[scale,:])
			std = np.std(region[scale,:])
			data = np.array([mean,std])
			features = np.vstack([features, data ]) if features.size else data
		
		r = img_rgb[py,px,0]
		g = img_rgb[py,px,1]
		b = img_rgb[py,px,2]

		color = np.mean(image_gray[py,px])
		r_color = np.mean(r)
		g_color = np.mean(g)
		b_color = np.mean(b)

		features = np.insert(features,0,color)
		features = np.insert(features,1,r_color)
		features = np.insert(features,2,g_color)
		features = np.insert(features,3,b_color)
		
		training_set = np.vstack([training_set, features ]) if training_set.size else features
		
	attr = ["color","r","g","b"]
	for i in range(t_scales):
		attr.append("mean_" + str(i))
		attr.append("std_" + str(i))

	df_training = pd.DataFrame(data = training_set)
	df_training['label'] = df['label'].values

	filename = dir_input.split('.tif')[0]
	df_training.to_csv(filename + "_training_set.csv",index=False)


def generate_training_set(group):
	n_features = 10
	dir_input = group['imagen'].iloc[0]
	dir_input = join('data',dir_input)
	if not(exists(dir_input)):
		return
	img_rgb = tiff.imread(dir_input)
	laplacians = get_laplacian(img_rgb,n_features)
	extract_data(img_rgb,laplacians,group,dir_input)
	
dir_filename = sys.argv[1]#regions_bird_detection.csv
data_regions = pd.read_csv(dir_filename)
data_regions.groupby('imagen').apply(generate_training_set)
