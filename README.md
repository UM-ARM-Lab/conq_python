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