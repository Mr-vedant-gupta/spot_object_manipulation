# spot_object_manipulation
Code base to enable Boston Dynamics Spot Robotics to detect, approach, and manipulate objects in indoor environments such as drawers and doors

The deep learning model for obejct recognition can be found here:
https://drive.google.com/drive/folders/1V2nNAiYEJJ25mDyB0mWGPy0UT8HQ8XsQ?usp=sharing (use handle-model, dogtoy-model is an older, less accurate version only for drawer handles)

Running the open drawer example:

Terminal window 1: python network_compute_server.py -m handle/exported-models/handle-model/saved_model handle/annotations/label_map.pbtxt 138.16.161.12

Terminal window 2: python fetch.py -s fetch-server -m handle-model -l {handle, door_handle} 138.16.161.12

upload graph to spot: python3 -m graph_nav_command_line --upload-filepath ~/drawer/navigation/maps/downloaded_graph 138.16.161.22

    
