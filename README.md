# face-landmark-localization
This is a project predict face landmarks (68 points) and head pose (3d pose, yaw,roll,pitch).


## Install
- [caffe](https://github.com/BVLC/caffe)
- [dlib face detector](http://dlib.net/)<p>
you can down [dlib18.17](http://pan.baidu.com/s/1gey9Wd1) <p>
cd your dlib folder<p>
cd python_example<p>
./compile_dlib_python_module.bat<p>
 add dlib.so to the python path<p>
if using dlib18.18, you can follow the [official instruction](http://dlib.net/)
- opencv<p>

## Usage for images

- Command : python landmarkPredict.py predictImage  testList.txt<p>
(testList.txt is a file contain the path of the images.)


## Usage for usb camera
This sctpt enables you to see intractive results for this face-landmark-localization.

usage: landmarkPredict_video.py uvcID


- Command :  python landmarkPredict_video.py  0

## Model

- You can download the pre-trained model from [here](http://pan.baidu.com/s/1c14aFyK)

## Train

- add train.prototxt and train_solver.prototxt files, Training using the 300W data set

## Result
![](result/1.png)
![](result/2.png)
![](result/3.png)

---
## class based rewrite for the landmarkPredict.
facePos.py: FacePosePredictor class is defined.

librect.py: helper functions for rectangles.

landmarkPredict_video.py uses this class version.
