# conq_python

This package is a no-ROS set of python libraries for using Conq, the Spot robot.

For running demos, user the ARM Razer laptop:

1. open a terminal
2. Connect to WiFi. You'll need the USB wifi dongle.

    ```bash
    nmcli con up spot-BD-21860002 
    ```

3. Launch pycharm
4. Launch the EStop GUI from the pycharm run configurations menu.
5. Launch one of the `_demo.py` scripts from the pycharm run configurations menu.

# Setup & installation

```
# Create virtual environment
virtualenv venv # give this a good name
source venv/bin/activate
git clone git@github.com:UM-ARM-Lab/conq_python.git
cd conq_python
pip install .  # installs in editable mode so your IDE knows where the conq_python packages are
```

## Additional setup for demos

Some demos have other dependencies that need to be setup. For example, for the regrasping demo, start with the general setup steps, then run:
```
# Install cdcpd_torch into some other folder (outside conq_python)
cd ..
git clone git@github.com:UM-ARM-Lab/cdcpd_torch.git -b conq_no_ros
cd cdcpd_torch
pip install . # inside cdcpd_torch
cd conq_python
pip install -e .[regrasping_demo]
```
