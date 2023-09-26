TODO

[] make CDCPD initialization better, possibly using the mask?
[] would be nice to improve the object searching functionality
    [x] try moving in a spiral pattern
    [x] we could also try looking along the length of the hose until we see an obstacle, but we'd still need a way to search for the hose in the first place
[] still having issues with MANIP_STATE_GRASP_PLANNING_NO_SOLUTION
  [] I think some of this is due to bad depth data. moving closer to the ground seems to help, but I'm not sure.
[] can we run CDCPD constantly somehow? 
[x] remove the return to home when stuck, instead just start searching for the obstacle/hose regrasp
[] integrate Alison's planning

DONE

[x] script dies if vacuum head is not detected
[x] need to re label instances for segmentation