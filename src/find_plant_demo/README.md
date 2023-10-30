<!--
Downloading, reproducing, distributing or otherwise using the SDK Software
is subject to the terms and conditions of the Boston Dynamics Software
Development Kit License (20191101-BDSDK-SL).
-->
# Demo Description
In this demo the robot will navigate to a desired location in the map, 
automatically detect a green object near its current location, and then pick up the object. 


## Setup Dependencies

These examples require the bosdyn API and client to be installed, and must be run using python3. Using pip, these dependencies can be installed using:

```
python3 -m pip install -r requirements.txt
```

This is based on a combination of the following examples: 
graph_nav_command_line (for walking to pre-determined location)
arm_grasp
(arm_simple)
(arm_walk_to_object)

# DEMO USAGE:
Before this demo can be run, you must generate a map of the environment as shown in the graph_nav_command_line example folder (in this case, the map is saved in ~/Downloads/collabspace2_simple_graph).

Get the ID of the waypoints you're interested in by the filenames in the folder
In my case: 
    Waypoint 0: south-cowrie-um.o6E6plYSbeaUK95peBw==
    Waypoint 45: unsaid-minnow-kMr5Fc65S2DYvMAwHol+rg==  (short code: um) (position [2.15, 3.39])

Also tune the HSV filter parameters using weed_detector/hsv_filter_tuning.py (make sure you run from within the weed_detector folder)
You can get pictures by running the get_image example with PIXEL_FORMAT_RGB_U8: 
    python3 get_image.py 192.168.80.3 --image-sources left_fisheye_image --pixel-format PIXEL_FORMAT_RGB_U8

Finally, update the HSV threshold values in weed_detector/base.py

python3 -m find_plant_demo --upload-filepath ~/Downloads/collabspace2_simple_graph 192.168.80.3
to upload the map
    5 [Enter]
To initialize at waypoint 0: 
    3 south-cowrie-um.o6E6plYSbeaUK95peBw== [Enter]
To navigate to Waypoint 45 and pickup the green object
    f um [Enter]

#TODO: Currently the robot will only look for green objects through its left camera, but it should look through all of them.
