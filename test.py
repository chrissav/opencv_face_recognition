import cv2
import numpy as np
import sqlite3
import os
import argparse

def runTest(testData='./testData'):
  imagePaths = [os.path.join(testData,f) for f in os.listdir(testData)]
  results = []
  (total, correct) = (0,0)

  for imagePath in imagePaths:
    if imagePath.endswith('.jpg'):
      img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
      faces = face_cascade.detectMultiScale(img, 1.3, 5)

      for (x,y,w,h) in faces:
        total += 1

        im = img[y:y+h,x:x+w]
        if resize:
            im = cv2.resize(im,(resize_size))
        ids,conf = recognizer.predict(im)
        c.execute("select name from users where id = (?);", (ids,))
        result = c.fetchall()
        name = result[0][0]

        if error_check:
          # get user id from imagePath.  requires test data to have user id in name.
          actual_name = imagePath.split('/')[-1].split('.')[0]
          if str(actual_name) == str(name):
            correct += 1
            print('match!', end=" ")
          else:
            print('error!', end=" ")
        print("Predicted User %s : %s" %(name, conf))

        results.append(conf)
  print('------------')
  print("Average confidence score: %s" % (sum(results) / (len(results))))
  if error_check:
    print("Average accuracy score: %s%%" %((correct / total) * 100))

def createRecognizer(t):
    rczr = None
    global resize
    if t.lower() == 'lbph':
        rczr =  cv2.face.LBPHFaceRecognizer_create()
        resize=False
    elif t.lower() == 'fisher':
        rczr = cv2.face.FisherFaceRecognizer_create()
        resize=True
    elif t.lower() == 'eigen':
        rczr =  cv2.face.EigenFaceRecognizer_create()
        resize=True
    elif t.lower() == 'bif':
        rczr = cv2.face.BIF_create()
    else:
        rczr =  cv2.face.LBPHFaceRecognizer_create()
        resize=False

    return rczr

def main():
  parser = argparse.ArgumentParser(description='Runs detector against testdata')
  parser.add_argument('--test-data',
    help="Path to test data directory.  Default: './testData'")
  parser.add_argument('--cascade-classifier',
    help="Path to cascade classifier to use.  Default: 'haarcascade_frontalface_default.xml'")
  parser.add_argument('--learning-algorithm',
      help="Learning Algorithm used. Options:'lbph','fisher','eigen','bif'. Default: 'lbph'")
  parser.add_argument('--resize-width',
      help="Integer which all facial images will be resized to. Only used if the learning algorithm needs it.")
  parser.add_argument('--resize-height',
      help="Integer which all facial images will be resized to. Only used if the learning algorithm needs it.")
  parser.add_argument('--error-check',
      help="Flag to run error checking",
      action='store_true')

  args = parser.parse_args()

  global error_check
  if args.error_check:
    error_check = True
  else:
    error_check = False

  width = 100
  if args.resize_width:
      width = args.resize_width

  height = 100
  if args.resize_height:
      width = args.resize_height

  global resize_size
  resize_size = (width,height)

  conn = sqlite3.connect('database.db')
  global c
  c = conn.cursor()

  fname = "recognizer/trainingData.yml"
  if not os.path.isfile(fname):
      print("Please train the data first")
      exit(0)

  #Cascade file
  classifierFile = './cascadeClassifiers/haarcascade_frontalface_default.xml'
  if args.cascade_classifier:
      classifierFile = args.cascade_classifier

  global face_cascade

  face_cascade = cv2.CascadeClassifier(classifierFile)

  global recognizer
  if args.learning_algorithm:
      recognizer = createRecognizer(args.learning_algorithm)
  else:
      recognizer = cv2.face.LBPHFaceRecognizer_create()

  recognizer.read(fname)

  if args.test_data:
    runTest(args.test_data)
  else:
    runTest()


if __name__ == "__main__":
    main()
