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
import argparse
import shutil


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
                if 'jpg' in name: #only want to use jpgs
                    people[os.path.basename(path)].append(os.path.join(path, name))

    return people

'''
NAME:       add_to_dataset
PURPOSE:    Takes in a list of names and adds them to database.
            Makes calls to save off grayscale images
INPUT:      Dict of names and paths.
RETURN:     0 if successful, 1 if error.
'''
def add_to_dataset(people, destinationPath):
    names = people.keys()

    try:
        connection = sqlite3.connect('database.db')
        cursor = connection.cursor()

        for name in names:
            if len(people[name]):
                cursor.execute('INSERT INTO users (name) VALUES (?)', (name,))
                uid = cursor.lastrowid
                img_num = 1
                for image in people[name]:
                    save_grayscale_img(image, 'User.' + str(uid) + '.' + str(img_num),destinationPath)
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
def save_grayscale_img(source, save_name, destinationPath):
    if destinationPath[-1] != '/':
        destinationPath = destinationPath + '/'

    grayscale = cv2.imread(source, cv2.IMREAD_GRAYSCALE)
    faces = face_cascade.detectMultiScale(grayscale, 1.3, 5)
    for (x,y,w,h) in faces:
        name = destinationPath  + save_name + '.jpg'
        im = grayscale[y:y+h,x:x+w]
        if resize:
            im = cv2.resize(im,resizeSize)
        cv2.imwrite(name, im)

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
    cv2.imshow("Training..." ,faceNp)

    cv2.waitKey(10)
  return np.array(IDs), faces

def train(path):
    Ids, faces = get_images_with_id(path)
    recognizer.train(faces,Ids)
    recognizer.save('recognizer/trainingData.yml')
    cv2.destroyAllWindows()


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

#---------------------------------------------------------------------------------------------------
# globals
#Recognizer used
recognizer = None
resizeSize = None
def main():
    parser = argparse.ArgumentParser(description='Runs detector against testdata')
    parser.add_argument('--training-data',
        help="Path to test data directory.  Default: './originalDataSet'")
    parser.add_argument('--dataset-path',
        help="Path to store the converted dataset  Default: './processedDataSet'")
    parser.add_argument('--cascade-classifier',
        help="Path to cascade classifier to use.  Default: 'haarcascade_frontalface_default.xml'")
    parser.add_argument('--learning-algorithm',
        help="Learning Algorithm used. Options:'lbph','fisher','eigen','bif'. Default: 'lbph'")
    parser.add_argument('--clear-data',
        help="Clears the data being run.")
    parser.add_argument('--resize-width',
        help="Integer which all facial images will be resized to. Only used if the learning algorithm needs it.", type=int)
    parser.add_argument('--resize-height',
        help="Integer which all facial images will be resized to. Only used if the learning algorithm needs it.", type=int)

    args = parser.parse_args()

    width = 100
    if args.resize_width:
        width = args.resize_width

    height = 100
    if args.resize_height:
        width = args.resize_height

    global resizeSize
    resizeSize = (width,height)


    global recognizer
    if args.learning_algorithm:
        recognizer = createRecognizer(args.learning_algorithm)
    else:
        global resize
        resize=False
        recognizer = cv2.face.LBPHFaceRecognizer_create()

    #Paths
    images_path = './originalDataSet'
    if args.training_data:
        images_path = args.training_data

    dataset_path = './processedDataSet'
    if args.dataset_path:
        dataset_path = args.dataset_path

    if args.clear_data:
        connection = sqlite3.connect('database.db')
        cursor = connection.cursor()
        cursor.execute('delete from users')
        connection.commit()
        connection.close()
        if os.path.exists(os.path.abspath(dataset_path)):
            shutil.rmtree(os.path.abspath(dataset_path))

    #Cascade file
    classifierFile = './cascadeClassifiers/haarcascade_frontalface_default.xml'
    if args.cascade_classifier:
        classifierFile = args.cascade_classifier

    classifierFile = os.path.abspath(classifierFile)
    if not os.path.exists(classifierFile):
        print('Could not find the correct classifier please check your path')
        raise ValueError('Could not find the correct classifier please check your path')

    global face_cascade
    face_cascade = cv2.CascadeClassifier(classifierFile)

    #Make sure we have the dataset directory if not create one
    if not os.path.exists(dataset_path):
        print('Creating dataset dir...')
        os.makedirs(dataset_path)

    #Make sure we have the recognizer directory if not create one
    if not os.path.exists('./recognizer'):
        print('Creating recognizer dir...')
        os.makedirs('./recognizer')

    #Get individuals and their images from root directory
    print('Checking provided images...')
    people = get_people(images_path)

    #Write the names to database and generate grayscale images
    print('Converting images to grayscale and adding them to dataset...')
    write = add_to_dataset(people,dataset_path)

    if write:
        print('Training...')
        train(dataset_path)
        print('Done!')
    else:
        print('ERROR: Failed to process data! Cannot train!')


if __name__ == "__main__":
    main()

