'''
DESCRIPTION:    This script will look into the images_path directory and expect a subdirectory for
                each user. The directory will be trated as the name. Contents of each subdirectory
                should be images of the individual. Each image will be converted to grayscale and
                copied over to dataset directory. Database will be populated with names.
'''


import os
import cv2
import numpy as np
from PIL import Image
import sqlite3

'''
NAME:       get_people
PURPOSE:    Returns a dict of subdirectories and their contents
INPUT:      Path to root directory of images.
RETURN:     Dict of paths.
'''
def get_people(path):
    
    people = {}
    
    for path, subdirs, files in os.walk(path):
        if not os.path.basename(path)is path:
            people[os.path.basename(path)] = []
            for name in files:
                people[os.path.basename(path)].append(os.path.join(path, name))

    return people

'''
NAME:       add_to_dataset
PURPOSE:    Takes in a list of names and adds them to database.
            Makes calls to save off grayscale images
INPUT:      Dict of names and paths.
RETURN:     0 if successful, 1 if error.
'''
def add_to_dataset(people):
    names = people.keys()
    
    try:
        connection = sqlite3.connect('database.db')
        cursor = connection.cursor()

        for name in names:
            cursor.execute('INSERT INTO users (name) VALUES (?)', (name,))
            uid = cursor.lastrowid
            img_num = 1
            for image in people[name]:
                save_grayscale_img(image, 'User.' + str(uid) + '.' + str(img_num))
                img_num = img_num + 1
                
            
        connection.commit()
        connection.close()
        
        return True
    except Exception as e:
        print(e)
        return False

'''
NAME:       save_grayscale_img
PURPOSE:    Takes in an image path and saves a grayscale copy to dataset
INPUT:      Path of image.
RETURN:     Nothing.
'''
def save_grayscale_img(source, save_name):
    grayscale = cv2.imread(source, cv2.IMREAD_GRAYSCALE)
    faces = face_cascade.detectMultiScale(grayscale, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.imwrite('dataset/' + save_name + '.jpg', grayscale[y:y+h,x:x+w])

def get_images_with_id(path):
  imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
  faces = []
  IDs = []
  for imagePath in imagePaths:
    faceImg = Image.open(imagePath).convert('L')
    faceNp = np.array(faceImg,'uint8')
    ID = int(os.path.split(imagePath)[-1].split('.')[1])
    faces.append(faceNp)
    IDs.append(ID)
    cv2.imshow("Training...",faceNp)
    cv2.waitKey(10)
  return np.array(IDs), faces

def train(path):
    Ids, faces = get_images_with_id(path)
    recognizer.train(faces,Ids)
    recognizer.save('recognizer/trainingData.yml')
    cv2.destroyAllWindows()



#---------------------------------------------------------------------------------------------------

#Paths
images_path = 'faces'
dataset_path = 'dataset'

#Recognizer used
recognizer = cv2.face.LBPHFaceRecognizer_create()

#Cascade file
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Make sure we have the dataset directory if not create one
if not os.path.exists('./dataset'):
    print('Creating dataset dir...')
    os.makedirs('./dataset')
    
#Make sure we have the recognizer directory if not create one
if not os.path.exists('./recognizer'):
    print('Creating recognizer dir...')
    os.makedirs('./recognizer')
    
#Get individuals and their images from root directory
print('Checking provided images...')
people = get_people(images_path)

#Write the names to database and generate grayscale images
print('Converting images to grayscale and adding them to dataset...')
write = add_to_dataset(people)

if write:
    print('Training...')
    train(dataset_path)
    print('Done!')
else:
    print('ERROR: Failed to process data! Cannot train!')
