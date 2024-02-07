# %% 
from matplotlib import pyplot as plt
import numpy as np
from scipy import linalg as la
from sklearn.cluster import KMeans
from imageio import imread, imsave
import cv2 as cv
import imutils
# %%
def quantize_image(im, n_clusters=4, res=200):
    """Cluster the pixels of the image 'im' by color.
    Return a copy of the image where each pixel is replaced by the value
    of its cluster center.
    
    Parameters:
        im ((m,n,3) ndarray): an image array.
        n_clusters (int): the number of k-means clusters.
        res (int): new width of image in pixels
    
    Returns:
        ((m,n,3) ndarray): the quantized image.
    """
    resized = imutils.resize(im, width=res)
    flat = resized.flatten().reshape(-1, 3)

    kmeans = KMeans(n_clusters=n_clusters).fit(flat)
    clusters = kmeans.predict(flat)
    centers = kmeans.cluster_centers_
    for i, center in enumerate(centers):
        flat[clusters == i] = center
        
    return flat.reshape(resized.shape)


# %%
fig, axs = plt.subplots(1, 4, figsize=(10,10))
picture = imread('images/cow.jpeg')
for i, q in enumerate([1, 6, 16]):
    new_pic = quantize_image(picture, q, 200)
    # print(len(picture[picture != new_pic]))
    axs[i%4].imshow(new_pic)
    axs[i%4].set_title(f"# of Clusters: {q}")
axs[3].imshow(imutils.resize(picture, width=100))
axs[3].set_title("Original")
plt.show()
# %%
q = 12

test = quantize_image(picture, q, 200)
plt.imshow(test)
plt.show()
kernel = np.ones((2,2),np.float32)/4
dst = cv.filter2D(test,-1,kernel)
test = quantize_image(dst, q)
plt.imshow(test)
plt.show()
colors = list(set([tuple(ii) for i in test for ii in i]))

for c in colors:
    temp = test.copy()
    mask = (temp != c).sum(axis=2)>0
    temp[mask] = np.array([0,0,0])
    plt.imshow(temp)
    plt.show()
# %%
im = cv.imread('images/cow.jpeg')
imgray = cv.cvtColor(test, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours, _ = cv.findContours(imgray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
plt.imshow(cv.drawContours(test, contours, -1, (0,255,0), 3))
# %%
plt.imshow(test)
# %%
im = plt.imread('images/cow.jpeg')



contours, _ = cv.findContours(im, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
plt.imshow(cv.drawContours(test, contours, -1, (0,255,0), 3))
# %%
im = plt.imread('images/cow.jpeg')
test = quantize_image(im, 12, 200)
test = imutils.resize(test, width=im.shape[1])
imgray = cv.cvtColor(test, cv.COLOR_RGB2GRAY)

contours, _ = cv.findContours(test, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
plt.imshow(cv.drawContours(im, contours, -1, (0,255,0), 3))
# %%
