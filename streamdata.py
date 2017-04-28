import pickle
import urllib.request
from xml.dom.minidom import parse
import random
import numpy as np
import time
import tensorflow as tf
import webbrowser


labelcount = 2
featuresize = 313

a_0 = tf.placeholder(tf.float32, shape=[None, featuresize])
y = tf.placeholder(tf.float32, shape=[None, labelcount])


middle = 2000
middle2 = 2000
middle3 = 1500
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
w_1 = weight_variable([featuresize, middle])
b_1 = bias_variable([1, middle])
w_2 = weight_variable([middle, middle2])
b_2 = bias_variable([1, middle2])
w_3 = weight_variable([middle2, middle3])
b_3 = bias_variable([1, middle3])
w_4 = weight_variable([middle3, labelcount])
b_4 = bias_variable([1, labelcount])
keep_prob1 = tf.placeholder(tf.float32)
keep_prob2 = tf.placeholder(tf.float32)
keep_prob3 = tf.placeholder(tf.float32)
keep_prob4 = tf.placeholder(tf.float32)
drop1 = tf.nn.dropout(w_1, keep_prob1)
drop2 = tf.nn.dropout(w_2, keep_prob2)
drop3 = tf.nn.dropout(w_3, keep_prob3)
drop4 = tf.nn.dropout(w_4, keep_prob4)

a_1 = tf.nn.relu(tf.add(tf.matmul(a_0, drop1), b_1))
a_2 = tf.nn.relu(tf.add(tf.matmul(a_1, drop2), b_2))
a_3 = tf.nn.relu(tf.add(tf.matmul(a_2, drop3), b_3))
a_4 = tf.add(tf.matmul(a_3, drop4), b_4)

acct_mat = tf.argmax(a_4, 1)
acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))
sess = tf.Session()
saver = tf.train.Saver()


searchdb ="./Data/searchterm.pk1"
searchesstore = open(searchdb, 'rb')
searches = pickle.load(searchesstore)
searchesstore.close()

wordmodel = "./Data/wordmodel.pk1"
wordmodeldump = open(wordmodel, 'rb')
model = pickle.load(wordmodeldump)
wordmodeldump.close()

filename = "./Data/model1.pk1" 
opener = open(filename, 'rb')
model1 = pickle.load(opener)
opener.close()

filename = "./Data/model2.pk1" 
opener = open(filename, 'rb')
model2 = pickle.load(opener)
opener.close()

filename = "./Data/model3.pk1" 
opener = open(filename, 'rb')
model3 = pickle.load(opener)
opener.close()

filename = "./Data/model4.pk1" 
opener = open(filename, 'rb')
model4 = pickle.load(opener)
opener.close()

filename = "./Data/model5.pk1" 
opener = open(filename, 'rb')
model5 = pickle.load(opener)
opener.close()

wordmodel = "./Data/wordmodel.pk1"
wordmodeldump = open(wordmodel, 'rb')
model = pickle.load(wordmodeldump)
wordmodeldump.close()

searchindex = 1
while searchindex != -1:
    searchindex = random.randrange(0,len(searches))
    search = searches[searchindex]
    print(search)
    url = 'http://svcs.ebay.com/services/search/FindingService/v1?OPERATION-NAME=findItemsByKeywords&sortOrder=StartTimeNewest&buyerPostalCode=92128&SERVICE-VERSION=1.13.0&SECURITY-APPNAME=RyanChes-EbaySear-PRD-d13d69895-95fa1322&RESPONSE-DATA-FORMAT=XML&REST-PAYLOAD&keywords=' + search
    url = url.replace(" ", "%20")
    apiResult = urllib.request.urlopen(url)
    document = apiResult
    parseddoc = parse(document)
    items = parseddoc.getElementsByTagName("item")
    x = 0
    for item in items:
        if item in items:
            features = []
            itemData = [0,1,2,3,4,5,6]
            a = np.array(())
            itemURL = items[x].getElementsByTagName("viewItemURL")[0].firstChild.data
            itemData[0] = items[x].getElementsByTagName("title")[0].firstChild.data.lower().split()
            itemData[1] = model1.transform([items[x].getElementsByTagName("categoryId")[0].firstChild.data]).tolist()[0]
            itemData[2] = model2.transform([items[x].getElementsByTagName("country")[0].firstChild.data]).tolist()[0]
            itemData[3] = model3.transform([items[x].getElementsByTagName("bestOfferEnabled")[0].firstChild.data]).tolist()[0]
            try:
                itemData[4] = model4.transform([items[x].getElementsByTagName("conditionId")[0].firstChild.data]).tolist()[0]
            except:
                print("condition not gathered")
                itemData[4] = [0,0,0,0,0,0,0]
                
            itemData[5] = model5.transform([items[x].getElementsByTagName("listingType")[0].firstChild.data]).tolist()[0]
            try:
                itemData[6] = [float(items[x].getElementsByTagName("shippingServiceCost")[0].firstChild.data)+ float(items[x].getElementsByTagName("convertedCurrentPrice")[0].firstChild.data)]
            except:
                itemData[6] = [float(100000)]
            
            for y in range(len(itemData[0])):
                try:
                    b = None
                    b = model[itemData[0][y]]
                except:
                    pass
                if b is not None:
                    a = np.hstack((a,b))
                else:
                    pass
            changesize = 215 - a.size
            try:
                a = np.pad(a,(0, changesize), 'constant', constant_values = (0))
                itemData[0] = a
            except:
                print("word vector was too long")
                itemData[0] = np.zeros(215)
            for z in itemData:
                for y in z:
                    features.append(y)
            features = np.array([features])
            x+=1
            with tf.Session() as sess:
              saver.restore(sess, "./Data/Modelv1")
              res = sess.run(acct_mat, feed_dict = {a_0: features, keep_prob1: 1.0, keep_prob2 : 1.0 ,keep_prob3: 1.0, keep_prob4:1.00})
              if res == [0]:
                print(res)
                webbrowser.open(itemURL, new=0, autoraise=True)
                time.sleep(15)
        else:
            pass

          
