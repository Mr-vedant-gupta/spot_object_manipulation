# spot_object_manipulation
Code base to enable Boston Dynamics Spot Robotics to detect, approach, and manipulate objects in indoor environments such as drawers and doors

The deep learning model for obejct recognition can be found here:
https://drive.google.com/drive/folders/1V2nNAiYEJJ25mDyB0mWGPy0UT8HQ8XsQ?usp=sharing

Running the open drawer example:
Terminal window 1: python fetch.py -s fetch-server -m dogtoy-model -l handle 138.16.161.12
Terminal window 2: python network_compute_server.py -m handle/exported-models/handle-model/saved_model -l handle/annotations/label_map.pbtxt 138.16.161.12
    
