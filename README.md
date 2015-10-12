Title : axarray 
Author : sylvain.guieu@gmail.com

# Introduction
axarray is a numpy array where axes can be labeled.

The idea is to be able to manipulate array, and do operation on axis without knowing the array shape order but on knowing labels related to the 'physical' meaning of the axes.

Often in science, it is useful to name the array axes by an intelligible label. 
For instance, for 2d images taken at different time, axes name of the obtain cube could be `["time", "y", "x"]`


axarray object aims to do that. For instance `a.mean(axis="time")` will execute  the mean on the axis labeled `"time"` where ever it is.

Given a1 and a2, two axarray, binary operation like a1+a2 can be performed even if the two axarray has different axes order as long as they have matching axis labels. 

# installation 
With pip
```
> pip install axarray 
```

Or from git in your PYTHON_PATH

```
> git clone https://github.com/SylvainGuieu/axarray.git
```

# Examples 

```python
>>> a = axarray( np.random.random((10,4,5)), ["time", "y", "x"])
>>> b = a.transpose( ["x","time", "y"])
>>> b.axes
["x","time", "y"]
```

can operate 2 transposed axarray as long as they match axis names 

```python
>>> (a+b).axes
["time", "y", "x"]
```
use the numpy frunction with axis labels

```python
>>> a.min(axis="time").shape
(4,5) 
# similar to: 
>>> np.min(a , axis="time")
```

axis can be alist of axis label

```python        
>>> a.mean(axis=["x","y"]).shape
(10,)
```        

one can use the convenient apply method. Useful in non-direct call as in a plot func for instance  

```python
>>> a.apply(time_reduce=np.mean, y_idx=slice(0,2)).shape
(2,5)
```

transpose, reshape rename axes in one call 

```python
>>> at = a.transform( [("pixel", "y","x"), "time"])        
>>> at.shape
(20, 10)  # (4*5, 10)
>>> at.axes
['pixel', 'time']
```

Extract a spectrum from image from named indices 

```python
### make some indices 
>>> iy, ix = axarray( np.indices( (3,4)), [0 ,"spatial", "freq"])
>>> ax[:,iy,ix].axes
['time', 'spatial', 'freq']
```