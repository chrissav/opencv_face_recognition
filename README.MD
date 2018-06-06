# Face recognition on OpenCV 3.4.0 

The full tutorial can be found on the link: http://www.python36.com/face-recognition-using-opencv-part-1/

## Requirements can be found in the file requirements.txt

## Files

1> create_database.py (used to create sqlite database)

2> record_face.py (Records the Name associated with the face)

3> trainer.py (used to train LBPHFaceRecognizer for face recognition)

4> detector.py (detects the face from previously trained data and fetches corresponding info from database) 

5> train_from_files.py (train using LBPH, eigen, or fisher on any giving dataset as input)

6> test.py (run tests on test data given as input.  Outputs the overall confidence score and accuracy of the test data)

## Our Contributions

### Setup

To use `train_from_files.py` and `test.py`, it's required to setup the training and test data directories and files like this:

The training data should have a parent directory for each different person and the pictures of each person should be labeled as `username.#.jpg` so that each filename is unique.
```
trainingdata_root/
--> username
----> username.1.jpg
----> username.2.jpg
...
...
----> username.20.jpg
--> anotheruser
----> anotheruser.1.jpg
----> anotheruser.2.jpg
...
...
----> anotheruser.20.jpg
```

The test data should have all test images in the same parent directory, and labeled similary to the trainingdata
```
testdata_root/
--> username.17.jpg
--> anotheruser.13.jpg
--> bobby.3.jpg
...
```

### Running the training and tests

1> train_from_files.py
Usage:
```
python train_from_files.py --training-data path/to/trainingdata --learning-algorithm eigen
```

This is used to train on any training set passed in as a parameter with one of the 3 learning algorithms (eigen, fisher, or lbph).  The images are converted to grayscale and resized if needed.  A training template is placed in recognizer/trainingData.yml.  Test data can be run against this template.

2> test.py
Usage:
```
python test.py --test-data data/testdata --learning-algorithm lbph --error-check
```

The learning algorithm passed in here should be the same algorithm used in the training template (trainingData.yml).  `--error-check` will calculate the % of tests matched correctly.  It's able to determine the correct match by checking the filename in the testdata.  


### Continuous Authentication

A method to evaluate user authentication continuously was added to `detector.py`.  The video feed will display the confidence score obtained from each prediction and an overall score to determine whether the user is the person they say they are.  The 2 scores will update continuously in the camera screen.  When the overall score drops below a certain threshold the text and face box will turn from green to red, indicating there may be an issue authenticating the user.
