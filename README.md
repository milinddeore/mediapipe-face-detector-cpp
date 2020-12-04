# mediapipe-face-detector-cpp:
Mediapipe face detector tflite model running, without using mediapipe framework, c++ implementation.

# Build standalone Tensorflow Lite library for Linux OR MacOS:
Using the instruction mentioned in the following [repo](https://github.com/milinddeore/TfLite-Standalone-build-Linux-MacOS). Such that your tensorflow lite library
is ready for your platform. I have tested it on MacOS, it should work perfectly on Linux too. 

# Build Face Detection project:
Standalone setup means you have all the `included` files at one place i.e. `3rdparty` folder and `libtensroflowlite.dylib`(for MacOS) OR `libtensroflowlite.so`
(for Linux) library ready for linking under `libs` folder. 
Run `make.sh` file to compile your project. `make.sh` file essentially, compile and link the project as:

I am also using opencv library so make sure all the includes and libraries are present on your machine: `pkg-config --cflags opencv` and `pkg-config --libs opencv`.

```
g++ -std=c++11 '-Wl,-rpath,$$ORIGIN/lib'  face_detect.cpp -o face_detector -L../3rdparty/libs -ltensorflowlite -I ../3rdparty/tensorflow/    
lite/tools/make/downloads/flatbuffers/include  -I ../3rdparty/tensorflow/lite/tools/make/downloads/absl  -I ../3rdparty/include -I ../       
3rdparty/ `pkg-config --cflags opencv` `pkg-config --libs opencv`  -lstdc++ -ldl -lpthread -lm -lz
```

# Run the project:
```
./face_detector
```
![ScreenShot](https://github.com/milinddeore/mediapipe-face-detector-cpp/blob/main/ScreenShot.png)
