import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import tifffile as tiff
import itertools
import math
from math import sqrt
from scipy.ndimage import filters
from skimage import data, color, feature, exposure
from skimage.color import rgb2gray
from os.path import isfile, join
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from skimage.filters.rank import maximum
from skimage.morphology import disk
from scipy.stats import zscore
from skimage import measure
import seaborn as sns
from sklearn.model_selection import KFold,StratifiedKFold,cross_val_score
from sklearn.metrics import jaccard_similarity_score
from sklearn.externals import joblib
from skimage.filters import rank
from scipy.ndimage import filters
from skimage.feature import hessian_matrix
from skimage.filters import threshold_otsu
from skimage.draw import circle
import matplotlib.patches as mpatches

def laplacian(img,sigma=5.0):
    Hyy, Hyx, Hxx = hessian_matrix(img, sigma, order = 'rc')
    return Hxx + Hyy

def view_results(img_hr_rgb,results):
	results = results.max() - results

	label_image = measure.label(results)

	fig, ax = plt.subplots(figsize=(10, 6))
	ax.imshow(img_hr_rgb,interpolation="nearest")

	for region in measure.regionprops(label_image):
		y,x = region.centroid
		r = 3*math.sqrt(2)
		c = plt.Circle((x, y), r, color='yellow', linewidth=2, fill=False)
		ax.add_patch(c)

	ax.set_axis_off()
	plt.tight_layout()
	#fig.savefig("birds_detected.png")
	plt.show()

def testing(pca,clf,img_testing,dir_testing):
	df = pd.read_csv(dir_testing + "_testing_set.csv")
	num_pixels = len(df.iloc[:,4:])
	#rows = int(df.iloc[num_pixels-1]['y'] + 1)
	#cols = int(df.iloc[num_pixels-1]['x'] + 1)
	data_r = pca.transform(df.iloc[:,4:].apply(zscore))
	n_components = pca.n_components_
	print "number of components ", n_components
	likelihoods = clf.predict_proba(data_r[:,0:n_components])
	labels_predicted = clf.predict(data_r[:,0:n_components])
	df['label'] = labels_predicted
	labels = ["shadow","bird","tree"]
	fig, ax = plt.subplots(figsize=(10, 6))
	ax.imshow(img_testing)
	ax.set_axis_off()
	
	for index, row in df.iterrows():
		label = row['label']
		min_row = row["min_row"]
		max_row = row["max_row"]
		min_col = row["min_col"]
		max_col = row["max_col"]
		w = max_col - min_col
		h = max_row - min_row
		likelihood = likelihoods[index]
		max_index  = np.argsort(likelihood)[::-1][0]
		score = likelihood[max_index]
		prob_label = labels[max_index]
		prob_label = labels_predicted[index]
		if score > 0.8:
			if prob_label == 'bird':
				c = plt.Rectangle((min_col,min_row),w,h,color='red', linewidth=2, fill=False)
				ax.add_patch(c)
			#else:
			#	c = plt.Rectangle((min_col,min_row),w,h,color='yellow', linewidth=2, fill=False)
			#	ax.add_patch(c)

	plt.tight_layout()
	plt.show()
	#result_image = np.zeros((rows,cols),dtype=np.uint8)
	#df['label'] = df['label'].apply(lambda x: labels[x])
	#np.float32(df['label'].values.reshape((rows,cols)))
	"""
	
	for index, row in df.iterrows():
		label = row['label']
		#print likelihoods[index],max_index
		
			x = int(row["x"])
			y = int(row["y"])
			result_image[y,x] = 255
	tiff.imsave("results.tif",result_image)
	"""
	
def get_laplacian(image_gray):
	scales = np.linspace(2.5, 6.5, num=10)
	laplacians = []

	for scale in scales:
		img_laplacian = -1.0*filters.gaussian_laplace(image_gray,scale)
		r = scale*math.sqrt(2)
		#print img_laplacian.max()
		#img_laplacian = 1.0*img_laplacian/img_laplacian.max()
		#img_laplacian_max = maximum(img_laplacian, disk(r))
		laplacians.append(img_laplacian)

	laplacians = np.array(laplacians)
	
	return laplacians

def change_label(label):
	label = labels[label]
	return label

def generate_testing_set(dir_input):
	img_hr_rgb = tiff.imread(dir_input + ".tiff")
	image_gray = rgb2gray(img_hr_rgb)
	laplacians = get_laplacian(image_gray)
	n_scales,rows,cols = laplacians.shape

	data = []
	for i,j in itertools.product(range(rows),range(cols)):
		positions = np.array([i,j])
		pixel_vector = laplacians[:,i,j]
		r = img_hr_rgb[i,j,0]
		g = img_hr_rgb[i,j,1]
		b = img_hr_rgb[i,j,2]
		color = image_gray[i,j]
		pixel_vector = np.hstack((positions,[color,r,g,b],pixel_vector))
		data.append(pixel_vector)
	#data = laplacians.reshape((rows*cols,n_scales))
	#data = np.column_stack((positions,data))
	attr = ["y","x","color","r","g","b"]
	for i in range(n_scales):
		attr.append("scale_" + str(i))
	df = pd.DataFrame(data = data,columns=attr)
	df.to_csv(dir_input + "_testing_set.csv",index=False)




def detect_blobs(img):
	"""
	t = 5
	img_laplacian = -1.0*filters.gaussian_laplace(img,t)
	blobs_log = []
	threshold_global_otsu = threshold_otsu(img_laplacian)
	img_thresholding = img_laplacian > threshold_global_otsu
	img_thresholding = img_thresholding*255

	img_dilation = rank.maximum(img_laplacian, morphology.disk(4))
	
	"""
	contours = []
	rows,cols = img.shape
	img_contours = np.zeros((rows,cols))
	blobs_log = feature.blob_log(img, min_sigma=2.5, max_sigma=4, num_sigma=20, threshold=.1)
	blobs_log[:, 2] = np.floor(blobs_log[:, 2] * sqrt(2))

	fig, ax = plt.subplots(figsize=(7, 7))
	ax.imshow(img, cmap=plt.cm.gray)

	for i,blob in enumerate(blobs_log):
		y, x, r = blob
		rr, cc = circle(y, x, r)
		c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
		ax.add_patch(c)
		#ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
		try:
			img_contours[rr, cc] = 1
		except Exception as e:
			print e

	ax.set_xticks([]), ax.set_yticks([])
	ax.axis([0, img.shape[1], img.shape[0], 0])
	plt.show()

	img_contours = np.float32(img_contours)
	tiff.imsave("blobs.tif",img_contours)
	label_image = measure.label(img_contours)
	
	for region in measure.regionprops(label_image):
		y,x = region.centroid
		(min_row, min_col, max_row, max_col) = region.bbox
		contours.append(region.bbox)
	
	contours = np.array(contours)
	
	return contours


from skimage.transform import pyramid_gaussian

labels = {
	"ArbolA": "tree",
	"ArbolB": "tree",
	"Sombra": "shadow",
	"juvenil": "bird",
	"Madera": "shadow",
	"None": "shadow",
	"hembra": "bird_w",
	"macho": "bird_m",
	"shadow" : 0,
	"bird" : 1,
	"tree": 2
}
import cv2
from scipy.misc import bytescale

if __name__ == '__main__':
	
	dir_testing = sys.argv[1]
	img_testing = tiff.imread(dir_testing)
	rows,cols,channels = img_testing.shape

	if channels > 3:
		img_testing = img_testing[:,:,0:3]
		r = bytescale(img_testing[:,:,0])
		g = bytescale(img_testing[:,:,1])
		b = bytescale(img_testing[:,:,2])
		img_testing = cv2.merge([r,g,b])

	print img_testing.shape

	image_gray = rgb2gray(img_testing)
	blobs = detect_blobs(image_gray)

	laplacians = get_laplacian(image_gray)
	t_scales,row,col = laplacians.shape
	testing_set = np.array([])

	for i,blob in enumerate(blobs):
		(min_row, min_col, max_row, max_col) = blob
		region_color = image_gray[min_row:max_row,min_col:max_col]

		r = np.mean(img_testing[min_row:max_row,min_col:max_col,0])
		g = np.mean(img_testing[min_row:max_row,min_col:max_col,1])
		b = np.mean(img_testing[min_row:max_row,min_col:max_col,2])
		region = laplacians[:,min_row:max_row,min_col:max_col]
		n_scales,rows,cols = region.shape
		color = np.mean(region_color)
		features = np.array([min_row,min_col,max_row,max_col,color,r,g,b])

		for scale in range(n_scales):
			mean = np.mean(region[scale,:,:])
			std = np.std(region[scale,:,:])
			data = np.array([mean,std])
			features = np.hstack([features, data ]) if features.size else data

		testing_set = np.vstack([testing_set, features ]) if testing_set.size else features

	attr = ["min_row","min_col","max_row","max_col","color","r","g","b"]
	for i in range(t_scales):
		attr.append("mean_" + str(i))
		attr.append("std_" + str(i))
	attr_str = ",".join(attr)
	df_testing = pd.DataFrame(data = testing_set,columns=attr)
	df_testing.to_csv(dir_testing  + "_testing_set.csv",index=False)
	#load img_testing
	#load models
	pca = joblib.load("models/pca.pkl")
	clf = joblib.load("models/svm.pkl")
	result = testing(pca,clf,img_testing,dir_testing)
	#view_results(image_gray,result)