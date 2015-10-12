"""axarray is a numpy array where axes can be labeled.

The idea is to be abble to manipulate array, and do operation on axis without knowing 
the array shape order but on knowing labels related to the 'phisical' meaning of the axes.

Often in science, it is usefull to name the array axes by an inteligible label. 
For instance, for 2d images taken at different time, axes name of the obtain 
cube could be `["time", "y", "x"]`


axarray object aims to do that. For instance `a.mean(axis="time")` will execute  
the mean on the axis labeled `"time"` where ever it is.

Given a1 and a2, two axarray, binarry operation like a1+a2 can be performed even 
if the two axarray has different axes order as long as they have matching axis labels. 

Function/Classes
---------------
axarray : create an axarray type 

isaxarray : isaxarray(a) -> True if isinstance(a, axarray)
axarrayconcat : concaten a list of axarray into one axarray 
size : same as np.size but with axis labels
islistofaxarray : check if list of axarray 

"""
from .base import *
