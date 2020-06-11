from googlesearch import search
import requests
from bs4 import BeautifulSoup
import numpy as np
import urllib.request as urllib2
import os
import re
from glob import glob
import cv2

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import pandas as pd
import seaborn as sn


searches = ['tree.jpg', 
            'pig.jpg',
            'dog.jpg',
            'flower.jpg',
            'cat.jpg',
            'skyclouds.jpg'] # cloud sometimes return cluster info shit

url_lists = [[], [], [], [], [], []]

url_limit = 100
search_count = 0
for a_search in searches:
    print(a_search)
    for result in search(a_search, tld="co.in", num=url_limit, stop=url_limit, pause=3):
        url_lists[search_count].append(result)
    
    search_count += 1

root_dir = os.getenv("HOME") + "/clf_data"

if os.path.exists(root_dir) == False:
    os.makedirs(root_dir)

directories = list(map(lambda i: root_dir + '/' + i.split('.jpg')[0], searches))

for a_dir in directories:
    if os.path.exists(a_dir) == False:
        os.makedirs(a_dir)

list_count = 0
for a_list in url_lists:
    print(list_count)
    image_iter = 0
    
    for url in a_list:
        print('\t' + url)
        if image_iter >= 100:
            break
        
        try:
            html = urllib2.urlopen(url)
            bs = BeautifulSoup(html, 'html.parser')
            
            images = bs.find_all('img', {'src':re.compile('.jpg')})
            
            if len(images) > 0:
                for image in images:
                    img_url = image['src']
                    
                    if len(img_url.split("https:")) > 1:
                        img_data = requests.get(img_url).content
                        if len(img_data) >= 5000:
                            filename = directories[list_count] + '/' + \
                                       searches[list_count].split('.jpg')[0] + \
                                       '_{0}.jpg'.format(image_iter)
                            with open(filename, 'wb') as handler:
                                handler.write(img_data)
                            
                            image_iter += 1
        except (urllib2.HTTPError, urllib2.URLError) as e:
            print("Received a Forbidden Error 403, or URL Open error")
    
    list_count += 1
"""
 collect all the images to check their dimensions and get the lowest dimension (aside from RGB)
"""
all_img_shapes = []
imgs = glob(root_dir + '/*/*.jpg') 

for i in imgs:
    tmp_img = cv2.imread(i)
    all_img_shapes.append(list(tmp_img.shape[0:2]))

lowest_dim = np.asarray(all_img_shapes).min()

"""
 resize the images to the lowest dimension of all the images to make sure
 that the input for the random forest classifier are all the same
"""

test_dir = os.getenv("HOME") + "/test_clf_imgs/"
if os.path.exists(test_dir) == False:
    os.makedirs(os.getenv("HOME") + "/test_clf_imgs/")

for i in imgs:
    print(i)
    tmp_img = cv2.imread(i)
    
    new_w, new_h = int(lowest_dim), int(lowest_dim)
    
    resized_img = cv2.resize(tmp_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # cv2.imwrite(test_dir + os.path.basename(i), resized_img)
    cv2.imwrite(i, resized_img)

""" 
 labels should be between 0 and 1 (not animal - animal) match labels with directories...
 directory order is:

 ['clf_data/tree',
  'clf_data/pig',
  'clf_data/dog',
  'clf_data/flower',
  'clf_data/cat',
  'clf_data/skyclouds']
"""
# class labels to match directories
label_data = [0, 1, 1, 0, 1, 0]

resized_imgs = glob(root_dir + '/*/*.jpg')
# set up empty lists for input data and label data
X = []
y = []

"""
 seeing how good of a classification can be achieved through random forest with only 
 histogrammed rgb values
"""
label_count = 0
for j in directories:
    print("Current Directory, Gathering Data: {0}".format(j))
    resized_imgs = glob(j + '/*.jpg')
    for k in resized_imgs:
        r_img = cv2.imread(k)
        
        input_data = []
        for l in range(r_img.shape[0]):
            hist = np.histogram(r_img[l].T[2].flatten(), bins=20)[1].tolist()
            input_data += hist
        
        X.append(input_data)
        y.append(label_data[label_count])
    
    label_count += 1

X = np.asarray(X)
y = np.asarray(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

rf_clf = RandomForestClassifier(n_estimators=200)
rf_clf.fit(X_train, y_train)

rf_preds = rf_clf.predict(X_test)

false_positive_rate, true_positive_rate, _ = roc_curve(y_test, rf_preds)
roc_area_under_the_curve = auc(false_positive_rate, true_positive_rate)

plt.figure(figsize=(6, 10))
plt.subplot(2, 1, 1)
plt.plot(false_positive_rate, true_positive_rate, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_area_under_the_curve)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for RF Clf')
plt.legend(loc="lower right")

plt.subplot(2, 1, 2)
confusion_matrix = pd.crosstab(y_test, rf_preds, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)
plt.show()




















