import timeit
print(timeit.timeit('''
import pickle
import urllib.request
from xml.dom.minidom import parse
import random
import numpy as np
import tensorflow as tf
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib




labelcount = 2
featuresize = 313

a_0 = tf.placeholder(tf.float32, shape=[None, featuresize])
y = tf.placeholder(tf.float32, shape=[None, labelcount])


middle = 100
middle2 = 100
middle3 = 100
middle4 = 100
middle5 = 100
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
w_4 = weight_variable([middle3, middle4])
b_4 = bias_variable([1, middle4])
w_5 = weight_variable([middle4, middle5])
b_5 = bias_variable([1, middle5])
w_6 = weight_variable([middle5, labelcount])
b_6 = bias_variable([1, labelcount])

a_1 = tf.nn.relu(tf.add(tf.matmul(a_0, w_1), b_1))
a_2 = tf.nn.relu(tf.add(tf.matmul(a_1, w_2), b_2))
a_3 = tf.nn.relu(tf.add(tf.matmul(a_2, w_3), b_3))
a_4 = tf.nn.relu(tf.add(tf.matmul(a_3, w_4), b_4))
a_5 = tf.nn.relu(tf.add(tf.matmul(a_4, w_5), b_5))
a_6 = tf.add(tf.matmul(a_5, w_6), b_6)

acct_mat = tf.argmax(a_6, 1)
acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))
sess = tf.Session()
saver = tf.train.Saver()


searchdb ="./Data1/searchterm.pk1"
searchesstore = open(searchdb, 'rb')
searches = pickle.load(searchesstore)
searchesstore.close()

wordmodel = "./Data1/wordmodel.pk1"
wordmodeldump = open(wordmodel, 'rb')
model = pickle.load(wordmodeldump)
wordmodeldump.close()

filename = "./Data1/model1.pk1" 
opener = open(filename, 'rb')
model1 = pickle.load(opener)
opener.close()

filename = "./Data1/model2.pk1" 
opener = open(filename, 'rb')
model2 = pickle.load(opener)
opener.close()

filename = "./Data1/model3.pk1" 
opener = open(filename, 'rb')
model3 = pickle.load(opener)
opener.close()

filename = "./Data1/model4.pk1" 
opener = open(filename, 'rb')
model4 = pickle.load(opener)
opener.close()

filename = "./Data1/model5.pk1" 
opener = open(filename, 'rb')
model5 = pickle.load(opener)
opener.close()

dbfile = "./Data/itemsAlerted.pk1" 
opener = open(dbfile, 'rb')
itemsAlerted = pickle.load(opener)
opener.close()

wordmodel = "./Data1/wordmodel.pk1"
wordmodeldump = open(wordmodel, 'rb')
model = pickle.load(wordmodeldump)
wordmodeldump.close()

me = "alerts@itconnected.tech"
you = "alerts@itconnected.tech"
msg = MIMEMultipart('alternative')
msg['From'] = me
msg['To'] = you

counter = 0
searchindex = 1
with tf.Session() as sess:
    saver.restore(sess, "./Data1/Modelv1")
    while counter != 50:
      counter+=1
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
      html = []
      for item in items:
          del msg['Subject']
          msg['Subject'] = search
          if item in items:
                  features = []
                  itemData = [0,1,2,3,4,5,6]
                  a = np.array(())
                  itemURL = items[x].getElementsByTagName("viewItemURL")[0].firstChild.data
                  title = items[x].getElementsByTagName("title")[0].firstChild.data
                  itemId = items[x].getElementsByTagName("itemId")[0].firstChild.data
                  itemData[0] = items[x].getElementsByTagName("title")[0].firstChild.data.lower().split()
                  itemData[1] = model1.transform([items[x].getElementsByTagName("categoryId")[0].firstChild.data]).tolist()[0]
                  itemData[2] = model2.transform([items[x].getElementsByTagName("country")[0].firstChild.data]).tolist()[0]
                  itemData[3] = model3.transform([items[x].getElementsByTagName("bestOfferEnabled")[0].firstChild.data]).tolist()[0]
                  bestOffer = items[x].getElementsByTagName("bestOfferEnabled")[0].firstChild.data
                  try:
                          condition = items[x].getElementsByTagName("conditionDisplayName")[0].firstChild.data
                          itemData[4] = model4.transform([items[x].getElementsByTagName("conditionId")[0].firstChild.data]).tolist()[0]
                  except:
                          print("condition not gathered")
                  itemData[4] = [0,0,0,0,0,0,0]
                  itemData[5] = model5.transform([items[x].getElementsByTagName("listingType")[0].firstChild.data]).tolist()[0]
                  listingType = items[x].getElementsByTagName("listingType")[0].firstChild.data
                  try:
                          currentPrice = float(items[x].getElementsByTagName("shippingServiceCost")[0].firstChild.data)+ float(items[x].getElementsByTagName("convertedCurrentPrice")[0].firstChild.data)
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
                  res = sess.run(acct_mat, feed_dict = {a_0: features})
                  if res == [0]:
                          if listingType != 'Auction':
                                  if itemId not in itemsAlerted:
                                          dbstore = open(dbfile, 'wb')
                                          itemsAlerted.append(itemId)
                                          pickle.dump(itemsAlerted, dbstore)
                                          dbstore.close()
                                          print(res)
                                          #webbrowser.open(itemURL, new=0, autoraise=True)
                                          message = """<html><head></head><body><h1> """ +" "+str(listingType)+" "+""" Listing</h1><h2><br> Listing Title: """ + str(title) + """</h2><br> Item URL: <a href=\"""" + str(itemURL) + "\"" + """>Ebay Link</a> <br> Price: """ + str(currentPrice) + """ <br> Best Offer Accepted: """ + str(bestOffer) + """<br>""" + str(condition) + """</p><br></body></html>"""
                                          html.append(message)
      try:
              messages = ""
              for message in html:
                      messages = str(messages) + str(message)
                      part2 = MIMEText(messages, 'html')
                      msg.set_payload(part2)
              if messages == "":
                      print("empty message blocked")
              else:
##                      Server = smtplib.SMTP("secureus24.sgcpanel.com", 587)
##                      password = 'ebayAlerts_1'
##                      Server.starttls()
##                      Server.login(me, password)
##                      Server.sendmail(me, you, msg.as_string())
                      del msg['Subject']
##                      Server.quit()
      except:
          print ('Email could not be resolved')
      else:
          pass''', number = 1))
