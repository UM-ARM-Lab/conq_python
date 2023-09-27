TODO

[] do the transform math so that it walks to put the _hand_ at the goal, not the body.
[] figure out whether rotating the images when trying to grasp in non-hand cameras is ok or not?
[] make it look at the hose correctly after walking up to it
[] would be nice to improve the object searching functionality
    [x] try moving in a spiral pattern
    [x] we could also try looking along the length of the hose
    [] Try a cardioid where we explicitly rotate the base
[] store the latest hose points in odom frame so we don't have to search from scratch every time
[] make CDCPD initialization better, possibly using the mask?

DONE

[x] use ALL cameras to look for objects
[x] script dies if vacuum head is not detected
[x] need to re label instances for segmentation
[x] still having issues with MANIP_STATE_GRASP_PLANNING_NO_SOLUTION
    - fix was just reboot the robot
[x] remove the return to home when stuck, instead just start searching for the obstacle/hose regrasp
[x] integrate Alison's planning

