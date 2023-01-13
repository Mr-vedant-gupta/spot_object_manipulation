# spot_object_manipulation
Code base to enable Boston Dynamics Spot Robotics to detect, approach, and manipulate objects in indoor environments such as drawers and doors

The deep learning model for obejct recognition can be found here:
https://drive.google.com/drive/folders/1V2nNAiYEJJ25mDyB0mWGPy0UT8HQ8XsQ?usp=sharing (use handle-model, dogtoy-model is an older, less accurate version only for drawer handles)

Training Commands:

python split_dataset.py --labels-dir handle/annotations/ --output-dir handle/annotations/ --ratio 0.9

python generate_tfrecord.py --xml_dir handle/annotations/train --image_dir handle/images --labels_path handle/annotations/label_map.pbtxt --output_path handle/annotations/train.record

python generate_tfrecord.py --xml_dir handle/annotations/test --image_dir handle/images --labels_path handle/annotations/label_map.pbtxt --output_path handle/annotations/test.record


python model_main_tf2.py --model_dir=handle/models/my_ssd_resnet50_v1_fpn --pipeline_config_path=handle/models/my_ssd_resnet50_v1_fpn/pipeline.config --num_train_steps=20000

tensorboard --logdir=handle/models --bind_all

CUDA_VISIBLE_DEVICES="-1" python model_main_tf2.py --model_dir=handle/models/my_ssd_resnet50_v1_fpn --pipeline_config_path=handle/models/my_ssd_resnet50_v1_fpn/pipeline.config --checkpoint_dir=handle/models/my_ssd_resnet50_v1_fpn

python exporter_main_v2.py --input_type image_tensor --pipeline_config_path handle/models/my_ssd_resnet50_v1_fpn/pipeline.config --trained_checkpoint_dir handle/models/my_ssd_resnet50_v1_fpn/ --output_directory handle/exported-models/handle-model

python eval.py -i handle/images -m handle/exported-models/handle-model/saved_model -l handle/annotations/label_map.pbtxt -o handle/output

Running the open drawer example:

Terminal window 1: python network_compute_server.py -m handle/exported-models/handle-model/saved_model handle/annotations/label_map.pbtxt 138.16.161.12

Terminal window 2: python fetch.py -s fetch-server -m handle-model -l {handle, door_handle} 138.16.161.12

upload graph to spot: python3 -m graph_nav_command_line --upload-filepath ~/drawer/navigation/maps/downloaded_graph 138.16.161.22

    
