TODO

[] store the latest hose points in odom frame so we don't have to search from scratch every time
[] make CDCPD initialization better, possibly using the mask?
[] figure out if it's possible to walk more slowly when doing WalkToPointInImage

DONE

[x] do the transform math so that it walks to put the _hand_ at the goal, not the body.
[x] figure out whether rotating the images when trying to grasp in non-hand cameras is ok or not?
    - Using autorotate seems to break WalkToObjectInImage
[x] use ALL cameras to look for objects
[x] script dies if vacuum head is not detected
[x] need to re label instances for segmentation
[x] still having issues with MANIP_STATE_GRASP_PLANNING_NO_SOLUTION
    - fix was just reboot the robot
[x] remove the return to home when stuck, instead just start searching for the obstacle/hose regrasp
[x] integrate Alison's planning

