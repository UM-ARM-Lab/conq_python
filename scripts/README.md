# Scripts

For description of what each script does and how to use it, see the comment at the top of the file.



# Documentation for behavior-cloning with VR Demonstrations and [Octo](https://github.com/octo-models/octo)

### Steps to collect, train, and evaluate new behaviors

1. Use the `vr_ros2_bridge` packages to collect demonstrations.
    1. Run the Unity project on Windows, the ros2_tcp_endpoint on Ubuntu, and the `generate_data_from_vr` script from `conq_python` on Ubuntu.
    2. This will result in a bunch of pkl files in the `data` folder  in `conq_python` in a `unsorted` folder. You need to make a `train/val` split by making folders called `train` and `val` and moving pkls from `unsourted` into the respective folders.
2. Generate the dataset.
    1. First, run `[preprocesser.py](http://preprocesser.py)` in `conq_hose_manipulation_dataset_builder` to convert the pkl files above into a new set of pkl files that live in the `conq_hose_manipulation/pkls` folder inside the builder repo. This stages lets you debug and visualize the dataset.
    2. You may want to increment the version number each time you modify the dataset. The version number is in the builder python script.
    3. The run `tfds build` in the terminal inside the `conq_hose_manipulation` folder. This will take many minutes to complete, and the result will be in an OXE compliant format. This step is pretty black-box and hard to debug, hence splitting the dataset creation into two steps.
3. Fine tune. This is done by running `fine_tune_conq_hose` in the [ARMLab fork of Octo](https://github.com/UM-ARM-Lab/octo/blob/main/scripts/finetune_conq_hose.py)https://github.com/UM-ARM-Lab/octo/blob/main/scripts/finetune_conq_hose.py.
