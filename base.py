#################################
# version 1.0
# Author S.Guieu
# History
# Todo:
#
#  - Add a possibility of masked_array
# 
from __future__ import print_function, division

import numpy as np

__all__ = ["islistofaxarray", "isaxarray", "axarrayconcat", 
            "size","size_of_shape", "axarray"
          ]
VERBOSE = 1

## 
# list of keyword suffix / function for te apply method 
# becarefull order maters a lot !!!
_apply_funclist = ["idx", "section", "reduce"] 


def asarray(a):                    
    return np.asarray(a)

def islistofaxarray(lst):
    """ Return True if all items are axarray object with the same axes names """
    first = True
    for axa in lst:
        if not isaxarray(axa): return False
        if first:
            axes = axa.axes
            first = False
        else:
            if axes!=axa.axes: return False
    return True
def isaxarray(a):
    """ True is this is array is a axarray """
    return isinstance( a, axarray)


def axarrayconcat(lst, axis=None, check=True):
    """ Convert a flat list of axarray to axarray 

    Args:
        lst (iterable) :  of axarray items 
        axis (axis label) :  the axis label along the dimension of lst. If no
            label is given a array is returned 
        check (Optiona[bool]) : if True check if all items are compatible axarray 
            (e.i same axis names). Default is True. If is not the case the axarray 
            returned will have only on axes labeled. 
            Turn off if you are sure of what is parsed and save some time. 
            If False it is assumed that all axarray are the same, axes are taken from 
            the first item. 

    Returns:
        axarray (or array if axis is None)

    Examples:
        # make a list of data 
        >>> exposures =  [axarray(np.random.random(4,3), ["y", "x"]) for i in range(5)]            

        >>> axa = axarrayconcat(exposures, "time")
        >>> axa.axes
        ["time", "y", "x"]
        >>> axa.shape
        [5, 4, 3]
    """
    if axis is None:
        return asarray(lst)
    if not len(lst):
        return axarray( asarray(lst), [axis])
    if check:
        if not islistofaxarray(lst):
            return axarray( [ asarray(data) for data in lst]  , [axis])
        else:
            return axarray(  [ asarray(data) for data in lst] , [axis]+list(lst[0].axes))

    return axarray( [ asarray(data) for data in lst] , [axis]+list(lst[0].axes))




def _broadcast_axis(sec, axis, axes1, array_axes, axes2, ia, i):
    # this cause a problem
    # np.random.random( (100, 256, 320))[ np.array([1,2,3]), np.array([[1,2,3],[1,2,3]]) ].shape
    #

    if isinstance(sec, slice):
        #case of slice conserve the axes as it is
        if array_axes is None:
            return sec, axes1+[axis], array_axes, axes2, ia
        else:
            return sec, axes1, array_axes, axes2+[axis], ia
    if isinstance(sec, (int,long)):
        #case of integer the axes is lost
        return sec, axes1, array_axes, axes2, ia+1 if ia is not None else None
    if isaxarray(sec):
        # Take the axes of axarray has new axes (array_axes)
        # If array_axes already exists check compatibility
        # and return empty list
        if array_axes is not None:
            if array_axes[2] and sec.axes != array_axes[0]:
                raise ValueError("axes mismatch: objects cannot be broadcast to a single axes name: %s!=%s"%(sec.axes,array_axes))
            array_axes = (array_axes[0] , max( array_axes[1], len(sec.shape)), True)
            if (i-ia)>1:
                # array indexing in a not contigus order
                return sec, [],  array_axes, axes1+axes2, i
            else:
                return sec, axes1, array_axes, axes2, i

        return sec, axes1, (sec.axes, len(sec.shape), True), axes2, i
    else:
        # Everything else should be a list or ndarray
        #
        sec = np.array( sec )
        if array_axes is not None:
            if array_axes[2]: #axes are given by a previous axarray
                array_axes = (array_axes[0] , max( array_axes[1], len(sec.shape)), True)
            else:
                array_axes = (array_axes[0]+[axis], max( array_axes[1], len(sec.shape)), False)

            if (i-ia)>1:
                # array indexing in a not contigus order
                return sec, [],  array_axes, axes1+axes2, i
            else:
                return sec, axes1,  array_axes, axes2, i
        return sec, axes1, ([axis],len(sec.shape), False) , axes2, i

def _broadcast_axes( sections, axes):
    """ new axes from a list of indexes (slice, integer, array, ...)

    a slice wiil conserve axis:    (slice(0,2),), ["x"] -> ["x"]

    a integer will loose the axis :   slice(0,2),3), ["x", "y"]  -> ["x"]
    a axarray index will rename axis : 
    a array like object will conserve axis if flat, or makes extra if multidimentional 
    """ 
    axis1  = []
    axis2  = []
    array_axes = None
    ia = None
    for i,(sec,axis) in enumerate(zip(sections, axes)):
        sec, axis1, array_axes, axis2,ia = _broadcast_axis( sec,axis,  axis1, array_axes, axis2, ia, i)

    array_axes = _reform_array_axes (*array_axes[0:2]) if array_axes is not None else []
    return axis1+array_axes+axis2
    return tuple(newsections), newaxes

def _reform_array_axes( axes, N):
    if N==len(axes): return axes
    if N>len(axes):
        if len(axes)>1:
            return [ (tuple(axes), i) for i in range(N) ]
        else:
            return [ (axes[0], i) for i in range(N) ]

    return [tuple(axes)]


def _decore_loose_axes(func):
    """ a decorator for np function that make no sens with axes names 
    e.g ravel, reshape, flatten, ...., more  ? 

    """
    def decored_loose_axes(axa, *args, **kwargs):
        return func(asarray( axa), *args, **kwargs)
    decored_loose_axes.__doc__ = "axarray : the call of this function will forget the axes labels\n\n"+func.__doc__
    return decored_loose_axes

def _decore_reduce_func(reduce_func):
    """ decorator for numpy function that reduce axis e.g. mean, std, min, max etc... """
    def decorated_reduce_func( axa, *args, **kwargs):
        # assume first of args is always axis, is that true ?
        if len(args) and "axis" in kwargs:
            raise TypeError("%s got multiple values for keyword argument 'axis'"%ufunc)
        axis = args[0] if len(args) else kwargs.pop("axis", None)
        return axa._reduce_axis( reduce_func, axis=axis, **kwargs)

    decorated_reduce_func.__doc__ = "axarray: apply the equivalent numpy function on given axis name(s).\n\n"+reduce_func.__doc__

    return decorated_reduce_func








def _prepare_op(left, right):
    """
    prepare two axarray objects for ufunc operations
    if both left and right are axarray array, the dimensions with the same
    axes label are used for the operations. The return object in this case
    will have the axes label of the object with the largest number of axes.
    This function prepare the left and right operators in this way
    """
    revers = False
    if len(left.axes)< len(right.axes):
        right, left = left, right
        revers = True

    if left.axes == right.axes[-len(left.axes):]:
        return left, right, left.axes, left.axes

    newrightaxes = [ a for a in left.axes if a in right.axes ]

    newleftaxes = []
    for n in left.axes:
        if not n in newrightaxes:
            newleftaxes.insert(0, n)
        else:
            newleftaxes.append(n)
    if revers:
        right.transpose(newrightaxes), left.transpose( newleftaxes ),newrightaxes, right.axes

    return left.transpose(newleftaxes), right.transpose( newrightaxes ), newleftaxes, left.axes

def size(A, axis=None):
    """ Same as numpy.size but axis can be a axis label and a list of axis label 

    Args:
        A (array-like) : the array or axarray 
        axis (Optional) : axis label or list of axis label
    Returns:
        s(int) : array size or partial size if axis is given 
    """
    if axis is None:
        return A.size
    axes = A.axes if isaxarray(A) else range(len(A.shape))
    if isinstance(axis, (tuple,list)):
        return reduce( lambda x,y: x*np.size(A,axis=axes.index(y)), axis, 1)
    return np.size(A, axis=axes.index(axis))

def size_of_shape( shape, axes, axis=None):
    """ return the axarray size from its shape tuple and a list of axes name

    Args:
        shape (tuple/list) : array shape
        axes  (list) : list of array axes name 
        axis (optional[string]) : the axis label on wich the size is returned

    Returns:
        s(int) : array size 

    Notes:
        >>> size_of_shape( a.shape, a.axes)
       # is iddentical  to 
        >>> size(a)

    """

    if axis is None:
        return reduce( lambda x,y: x*y, shape, 1)
    if isinstance( axis, (tuple,list)):
        return reduce( lambda x,y: x*shape[axes.index(y)], axis, 1)
    return shape[axes.index(axis)]

def __lop__(op):
    """ binary left operator decorator """
    def tmpop(self,right):
        return self._run_op(self, right, op)
    return tmpop
def __rop__(op):
    """ binary right operator decorator """
    def tmpop(self,left):
        return self._run_op(left, self, op)
    return tmpop
def __uop__(op):
    """ unary operator decorator """
    def tmpop(self):
        return self.__class__(op(asarray(self)), list(self.axes))
    return tmpop


class axarray(np.ndarray):
    """ axarray is a numpy array with labeled axes 

    Often in science, it is usefull to name the array axes by an inteligible label. 
    For instance, for 2d images taken at different time, axes name of the obtain
    cube could be ["time", "y", "x"]

    axarray object aims to do that. Basic operations can be done without knowing
    the structure of the array. For instance  a.mean(axis="time") will execute 
    the mean on the axis labeled "time" where ever it is.

    Given a1 and a2, two axarray, binarry operation like a1+a2 can be performed 
    even if the two axarray has different axes order as long as they have matching 
    axis labels. 

    Args:
        A (array like) : 
        axes (iterable) : list of axis labels, can be any object, but string are 
            the most obvious.

        aliases (Optional[dict]) : An optional dictionary that define axis aliases     
            used *ONLY* in the apply method (this may change).
            for instance  aliases = {"pix": ["y", "x"]}
            will replace pix_reduce =f  by y_reduce = f, x_reduce= f
            and  pix_idx = (iy,ix)  by y_idx = iy, y_idx = ix
    Return:
        axarray instance

    Properies:
        A : return the array as a regular numpy array 
       + all other numpy array properties

    Attributes:
        axis_index :  -> a.axes.index(label)
        axis_len : -> a.shape[ a.axis_index(label) ]
        idx : select array indexes on given axis
        section : select array indexes one by one on given axis 
        transform : make a transpose and reshape in one command 
        reduce : reduce axis from a given reduce function (e.g. np.mean)
      + all other numpy array properties 
      
     Examples:
        >>> a = axarray( np.random.random((10,4,5)), ["time", "y", "x"])
        >>> b = a.transpose( ["x","time", "y"])
        >>> b.axes
        ["x","time", "y"]

        ## can operate 2 transposed axarray as long as they 
        ## match axis names 
        >>> (a+b).axes
        ["time", "y", "x"]

        ## use the numpy frunction with axis labels
        >>> a.min(axis="time").shape
        (4,5) 
        # similar to: 
        >>> np.min(a , axis="time") 
        
        # axis can be alist of axis label        
        >>> a.mean(axis=["x","y"]).shape
        (10,)
        
        # one can use the conveniant apply method. Usefull in non-direct
        # call as in a plot func for instance  
        >>> a.apply(time_reduce=np.mean, y_idx=slice(0,2)).shape
        (2,5)

        # transpose, reshape rename axes in one call 
        >>> at = a.transform( [("pixel", "y","x"), "time"])        
        >>> at.shape
        (20, 10)  # (4*5, 10)
        >>> at.axes
        ['pixel', 'time']

        ### e.g. extract a spectrum from image from named indices 
        ### make some indices 
        >>> iy, ix = axarray( np.indices( (3,4)), [0 ,"spatial", "freq"])
        >>> ax[:,iy,ix].axes
        ['time', 'spatial', 'freq']
                

    """
    _verbose = None # if None use the VERBOSE module default    
    apply_aliases = None
    def __new__(subtype, data_array, axes=None, aliases=None):
        # Create the ndarray instance of axarray type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of axarray.
        # It also triggers a call to InfoArray.__array_finalize__

        obj = asarray(data_array).view(subtype)

        if axes is None:
            # default axes are [0,1,2,3, ...]
            axes = range(len(obj.shape))
        elif isinstance(axes, str):            
            axes = [axes,]

        elif len(axes)>len(obj.shape):
            raise KeyError("len of axes must be inferior or equal to the array shape, got %d < %d"%(len(axes), len(obj.shape)) )
        if len(set(axes))!=len(axes):
            raise KeyError("All axes labels must be unique")
        obj.axes = axes
        if aliases:
            obj.apply_aliases = dict(aliases)
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(axarray, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. axarray():
        #    obj is None
        #    (we're in the middle of the axarray.__new__
        #    constructor, and self.info will be set when we return to
        #    axarray.__new__)
        if obj is None: return
        # From view casting - e.g arr.view(axarray):
        #    obj is arr
        #    (type(obj) can be axarray)
        # From new-from-template - e.g infoarr[:3]
        #    type(obj) is axarray
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'info', because this
        # method sees all creation of default objects - with the
        # axarray.__new__ constructor, but also with
        # arr.view(InfoArray).
        self.axes = getattr(obj, 'axes', range(len(obj.shape)))
        # We do not need to return anything

    @property
    def A(self):
        return asarray(self)


    def axis_len(self, axis):
        """
        a.axis_len(axis) -> len of the given axis
        if axis is None return a.size
        """
        if axis is None:
            return self.size
        return self.shape[self.axis_index(axis)]

    def axis_index(self, axislabel):
        """ A.axes.index(axis) """
        return self.axes.index(axislabel)
  

    def get_missing_axes(self, lst):
        """ from a list of axis label return a lis tof missing axis in the axarray """
        return [ axis for axis in self.axes if axis not in lst]

    def _get_axes(self, lst=None, alternate=None):
        """ """

        if alternate is None:
            alternate = self.axes


        if lst is None: return list(self.axes)
        lst = list(lst)
        cNone = lst.count(None)

        if cNone:
            if cNone>1:
                raise ValueError("Axes list allows only one None value")
            iNone = lst.index(None)
            lst.remove(None)
            lst = lst[0:iNone]+self.get_missing_axes(lst)+lst[iNone:]

        for axis in lst:
            if not axis in self.axes:
                return ValueError("wrong axis label %s"%axis)
        return lst

    def __array_wrap__(self, out_arr, context=None):
        #print 'In __array_wrap__:'
        #print '   self is %s' % repr(self)
        #print '   arr is %s' % repr(out_arr)
        # then just call the parent
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def __array_prepare__(self, obj, context=None):
        #if context:
        #    if isinstance(obj, axarray):
        #        left, right, axes, oaxes = _prepare_op(self, obj)
        #print obj.shape
        #print "context ", context
        return obj

    def _allsection(self):
        return [ slice(0,None) for a in self.axes ]

    def _section(self, allsection, axesname, section):
        allsection[self.axis_index(axesname)] = section
        if isinstance(section, int):
            return False
        return True

    def idx(self, section, axis=None):
        ###
        # To be optimized !!!!!
        # For a axarray
        # %timeit data[i]
        # 10000 loops, best of 3: 38.5 us per loop
        # For a regular array
        # In [622]: %timeit data.A[ia]
        # 100000 loops, best of 3: 4.36 us per loop
        # c
        if not isinstance(section,tuple):
            raise ValueError("first argument, section, must be a tuple, got a %s"%type(section))
        if axis is None:
            return self[section]

        N = len(section)
        if len(axis) != N:
            raise ValueError("axis keyword should have the same len than section tuple")
        if not N:
            return self
        if len(set(axis))!=len(axis):
            raise ValueError("All element of the axis list must be unique got %s"%axis)

        axes_index = [self.axis_index(ax) for ax in axis]
        N = max(axes_index)
        # build an empty list of section according to the max index
        allsection = [slice(0,None)]*(N+1)
        for sec,i in zip(section, axes_index):
            allsection[i] = sec
        return self[tuple(allsection)]


    def section(self, section, axis=None):
        allsection = self._allsection()
        allsection[self.axis_index(axis)] = section


        axes = list( self.axes) # copy of axes
        arout = asarray(self)[tuple(allsection)]

        if len(arout.shape)<len(self.shape):
            axes.remove(axis)
        if len(arout.shape)>len(self.shape):
            i = axes.index(axis)
            axes.remove(axis)
            # build a list of [(axis,num)] for axis that has been extended
            axes = axes[0:i]+[(axis,i) for i in range(len(arout.shape)-len(self.shape)+1)]+axes[i+1:]

            #axes = axes[0:i]+[axis]*( len(arout.shape)-len(self.shape))+axes[i:]
        return self.__class__( asarray(self)[tuple(allsection)], axes )

    def section_s(self, axesname_sec):
        # axesname_sec is a list of tuple [(axesname1, sec1), (axesname2, sec2), ....]
        if not len(axesname_sec): return self
        if isinstance( axesname_sec, dict):
            axesname_sec = axesname_sec.iteritems()

        allsection = self._allsection()
        axes = list( self.axes) # copy of axes
        for axesname, section in axesname_sec:
            if not self._section( allsection, axesname, section):
                axes.remove(axesname)
        return self.__class__( asarray(self)[tuple(allsection)], axes )

    def __str__(self):
        return "%s(\n%s,\naxes=%s)"%(self.__class__.__name__, str(asarray(self)), str(self.axes))
    def __repr__(self):
        return "%s(\n%s,\naxes=%s)"%(self.__class__.__name__, asarray(self).__repr__(), str(self.axes))

    def __getitem__(self, items):
        if not isinstance(items, tuple):
            items = (items,)
        Naxes = len(self.axes)
        Nitem = len(items)
        newaxis = _broadcast_axes(items, self.axes)+self.axes[Nitem:]

        if len(newaxis):
             return self.__class__(asarray(self)[items], newaxis)
        return asarray(self)[items]


    def _get_verbose(self):
        """
        return self._verbose or VERBOSE if None
        """
        return self._verbose if self._verbose is not None else VERBOSE
    @classmethod
    def _run_op(cls, left, right, op):
        if isaxarray(right) and isaxarray(left):
            left, right, axes, oaxes = _prepare_op(left, right)
            return cls(  op( asarray(left), asarray(right)), axes).transpose( oaxes)
        if isaxarray(left):
            return cls(  op(asarray(left), right) , list(left.axes))
        if isaxarray(right):
            return cls(  op(left, asarray( right)) , list(right.axes))
        return cls( op( left, right) )

    def _prepare_transform(self, reshapes, ignore_unknown):
        """
        From a list of axes name and formulae return a tuple containing:
        a list to pass to tranpose, a list new pass to a reshape and a list of new axes name.
        """
        datashape = self.shape
        newshape   = []
        transposes = []
        newaxes    = []
        allkeys = []

        for i,x in enumerate(list(reshapes)): #list make a copy
            if isinstance(x, tuple):
                if len(x)<2:
                    raise ValueError( "If axes def is tuple must be of len>1")

                if ignore_unknown:
                    x2 = list(x[0:1])+[k for k in x[1:] if (k in self.axes) or (k is None)]
                    if len(x2)>1:
                        reshapes[i] = tuple(x2)
                    else:
                        reshapes.remove(x)
                        i -= 1
                else:
                    x2 = x
                allkeys.extend(x2[1:])
            else:
                if ignore_unknown and (not x in self.axes) and (x is not None):
                    reshapes.remove(x)
                    i -= 1
                else:
                    allkeys.append(x)

        if allkeys.count(None):
            if allkeys.count(None)>1:
                raise ValueError( "None appears more than ones")
            allkeys.remove(None)


        if None in reshapes:
            iNone = reshapes.index(None)
            reshapes.remove(None)
            reshapes = reshapes[0:iNone]+self.get_missing_axes(allkeys)+reshapes[iNone:]

        for x in reshapes:
            if isinstance(x, tuple):
                if len(x)<2:
                    raise ValueError( "If axes def is tuple must be of len>1")
                newname = x[0]
                merged_axis = list(x[1:])
                if None in merged_axis:
                    iNone = merged_axis.index(None)
                    merged_axis.remove(None)
                    merged_axis =  merged_axis[0:iNone]+self.get_missing_axes(allkeys)+merged_axis[iNone:]

                indexes = [ self.axis_index(s) for s in merged_axis ]
                transposes.extend(merged_axis)
                newshape.append(reduce(lambda x, y: x*y, [datashape[i] for i in indexes]) )
                newaxes.append(newname)

            else:
                if x in self.axes:
                    i = self.axis_index(x)
                    transposes.append( x )
                    newaxes.append( x )
                    newshape.append( datashape[i] )
                else:
                    transposes.append( x )
                    newaxes.append( x )
                    newshape.append( 1 )


        return tuple(transposes), tuple(newshape), newaxes


    def apply(self, **kwargs):
        """ conveniant function to apply indexes and reducing in one command line 

        The goal is to quickly apply indexes and reducing without knowing the structure of the
        array but only the axis names. This function is for convenient use so is a bit durty.  

        Args:
            **kwargs :  keyword pair / vaules  can be 
                {axes_name}_idx  =  valid index for given axis (e.i., int, slice, array like)
                {axes_name}_section = valid index. The difference between _idx and _section is that:
                    apply(x_idx = ix,  y_idx = iy) -> will call A[iy,ix]  (if "y"/"x" is name of axis 0/1)
                    apply(x_section=ix , y_section=iy) -> will call A[iy] then index ix with the result.

                    All the *_idx, if array must be of the same dimension but not with *_section 
                    

                {axes_name}_reduce = f  reduce function with signature f(A, axis=axes_name) with A the array 
                    and axis the axis names

                *_idx are called first then *_section then *_reduce     

                Note: shorter version exists *_i for *_idx, *_s for *_section *_r for *_reduce       

            squeeze (Optiona[bool]): squeeze the returned axarray (remove axis with len 1) if True. default False
            aliases (Optional[dict]): A dictionary of aliases, for instance if aliases = {"pix": ("y", "x")} then 
                pix_idx = (4,5) keyword   will be replaced by  y_idx=4, x_idx=5  
                pix_reduce = np.mean      will be replaced by  y_reduce = np.mean, x_reduce = np.mean 
                the aliases keyword update the apply_aliases attribute of the axarray object (if any).


                
        Returns:
            A scalar or axarray

        Warning:
            All keywords that does not match {axis_name}_idx/section/reduce will be 
            totaly ignored silently.

        Examples:
            >>> from axarray import axarray
            >>> a = axarray( np.random.random(5,10,20), ["time", "y", "x])
            >>> a.apply(time_reduce=np.mean,  x_id=np.s_[0:2])
            >>> a.apply( time_r = np.mean)  # time_r is a short version of time_reduce   

            >>> a.apply( x_idx=[0,1,2,3], y_idx=[0,1,2,3]).shape
            (4,)
            >>> a.apply( x_section=[0,1,2,3], y_section=[0,1,2,3]).shape
            (4,4)

            >>> a = axarray( np.random.random(5,10,20), ["time", "y", "x],  {"pix":["y", "x"]})
            # make a function that return a portion of image
            >>> def mybox(pos=(0,0),size=10):
                return (slice( pos[0], pos[0]+size), slice( pos[0], pos[0]+size) )
            >>> a.apply( pix_idx = mybox(size=5) ) # pix_idx is alias of y_idx, x_idx     

        """
        squeeze = kwargs.pop("squeeze", False)
        aliases = self.apply_aliases or {}
        aliases.update(kwargs.pop("aliases", {}))

        ## 
        # remove the None, they are not relevant for 
        # any of the methods 
        for k,v in kwargs.items():
            if v is None: kwargs.pop(k)


        verbose = self._get_verbose()
        Nfunc = len(_apply_funclist)
        ifunc = 0
        #for funckey in self._apply_funclist:
        while ifunc<Nfunc:
            funckey = _apply_funclist[ifunc]
            axes = list(self.axes)

            f = funckey
            args    = {}
            notargs = []


            for ax in self.axes:
                lazykwargs = "%s_%s"%(ax,funckey)
                shortlazykwargs = "%s_%s"%(ax, funckey[0])
                if lazykwargs in kwargs and shortlazykwargs in kwargs:
                    raise ValueError("'%s' and '%s' keywords are the same use only one"%(lazykwargs, shortlazykwargs))                
                func_val = kwargs.pop(lazykwargs, kwargs.pop(shortlazykwargs, None))

                if func_val is not None:
                    args[ax] = func_val

            ## handle the aliases         

            for alias, al_axes in aliases.iteritems():
                lazykwargs = "%s_%s"%(alias, funckey)
                shortlazykwargs = "%s_%s"%(alias, funckey[0])
                if lazykwargs in kwargs and shortlazykwargs in kwargs:
                    raise ValueError("'%s' and '%s' keywords are the same use only one"%(lazykwargs, shortlazykwargs))                

                func_val = kwargs.pop(lazykwargs, kwargs.pop(shortlazykwargs, None))

                if func_val is not None:
                    if f in ["idx", "section"]:
                        ###
                        #  if pix alias of ["y", "x"]
                        # then  pix_idx = (4,5) ->  y_idx=4, x_idx=5
                        if not hasattr(func_val, "__iter__"):
                            func_val = [func_val]
                        for ax,v in zip(al_axes, func_val):
                            if ax in args and args[ax] != v:
                                raise TypeError("'%s' alias keyword in conflict with the '%s_%s' keyword"%(lazykwargs, ax, f))
                            args[ax] = v
                    else:
                        ###
                        # if pix alias of ["y", "x"]
                        # then  pix_reduce = mean ->  y_idx=mean, x_idx=mean    
                        for ax in al_axes:
                            if ax in args and args[ax] != v:
                                raise TypeError("'%s' alias  keyword in conflict with the '%s_%s' keyword"%(lazykwargs, ax, f))
                            args[ax] = func_val
                   

            if not len(args):
                # if there is no fancy keyword in kwargs go to the next func
                # because the self.axes can have changed we need to continue
                # until there is no fancy keyword available
                ifunc += 1
                continue

            if funckey== "idx":
                ###
                # for idx we need to collect them before
                indexes = []
                indexes_axis = []
                for ax in axes:
                    if ax in args and ax in self.axes:
                        indexes.append(args[ax])
                        indexes_axis.append(ax)
                self = self.idx(tuple(indexes), axis=indexes_axis)
                ########
                # At this point self can be something else
                # like a float for instance
                if not isaxarray(self): return self
            else:
                for ax in axes:

                    if ax in args and ax in self.axes:
                        self = getattr(self, f)( args[ax], axis=ax )
                        ########
                        # At this point self can be something else
                        # like a float for instance
                        if not isaxarray( self): return self


        
        # for k in kwargs:
        #     for f in _apply_funclist:
        #     if k.endswith("_"+f):
        #         raise TypeError("got '%s' keyword but the axis is unknown")        

        if verbose>=2 and len(kwargs):
            print ("NOTICE : keys %s had no effect on data"%kwargs.keys())

        if squeeze:
            return self.squeeze()
        return self



    def transform(self, reshapes, add_unknown=False, ignore_unknown=False,
                  squeeze_missing=True
                  ):
        """ make a transpose, reshape, rename of axes in one call

        
        Transform the array axes according to the axes list of axes label or tuple.
        transform can make a transpose, reshape and rename axes in the same call.

        Args:
            reshapes (list): list of axis labels that define the tranform    
                If item is a tuple, the tuple  first item should be the new axes label,
                the othes the label of existing axis
                >>> a = axarray( np.zeros((10,4,5)), ["time", "y", "x"])
                >>> a.transform( [("pix", "y", "x"), "time"])
             is equivalent to do
                >>> axarray( a.transpose( ["y","x","time"]).reshape( (4*5,10)), ["pix","time"] )  


            add_unknown (Optional[bool]): if  True unkwonw axes label will be added
                with a dimension of 1. Default is False 
            ignore_unknown (Optional[bool]): if True, unknown axis will be ignores.
                default is False. add_unknown, ignore_unknown can't be both True
            squeeze_missing (Optional[bool]): if True missing axis with dimension 1
                will be dropped. Else raise ValueError. Default is True    
                    
        If ignore_unknown is True, unknown axes will be completely ignored in
                           in the process.


        e.g.:

        > d = axarray(np.random.rand( 8, 10, 12,13), ["time", "z", "y", "x"])

        > d.transform( [ "z", ("pixel", "x", "y"), "time"] )
        axarray(([[...,
               [ 0.82653106,  0.99736293,  0.67030048, ...,  0.91404063,
                 0.71512099,  0.20758938]]]),('z', 'pixel', 'time'))
        # Equivalent to :
        > axarray ( d.transpose( [1,3,2,0] ).reshape( ( 10,13*12,8) ), ["z", "pixel", "time"])

        """
        if add_unknown is True and ignore_unknown is True:
            raise KeyError("add_unknown and ignore_unknown cannot be both True")
        transposes, newshape, newaxes = self._prepare_transform(reshapes, ignore_unknown)
        data = self.transpose(transposes, add_unknown=add_unknown,
                              squeeze_missing=squeeze_missing).reshape( newshape )
        return self.__class__( data, newaxes )

    @property
    def T(self):
        return self.transpose()

    def transpose(self, tps=None, add_unknown=False,
                  squeeze_missing=True,
                  **kwargs):
        if tps is None:
            tps = list(self.axes)
            tps[0], tps[-1] = tps[-1], tps[0] ## swap first and last axes

        axes = self.axes
        reshape_uk = False
        if add_unknown:
            newshape = []
            newtps  = []
            for tp in tps:
                if tp in axes:
                    newshape.append(self.axis_len(tp))
                    newtps.append(tp)
                else:
                    newshape.append(1)
                    reshape_uk = True
        else:
            for a in tps:
                if a not in axes:
                    raise TypeError("unknown axis '%s'"%a)

        reshape_missing = False
        if squeeze_missing:
            newshape_missing = []
            newaxes_missing = []
            for x in axes:
                if x not in tps:
                    if self.axis_len(x)>1:
                        raise ValueError("Cannot squeeze axis '%s', its len should be of size 1, got size %d"%(x,self.axis_len(x)))
                    reshape_missing = True
                else:
                    newshape_missing.append(self.axis_len(x))
                    newaxes_missing.append(x)
        if reshape_missing:
            self = self.__class__(self.reshape(newshape_missing),
                                  newaxes_missing)



        if reshape_uk:
            return self.__class__( asarray(self).transpose( [self.axis_index(tp) for tp in newtps ]).reshape(newshape), list(tps))

        return self.__class__(  asarray(self).transpose( [self.axis_index(tp) for tp in tps ]), list(tps))
    transpose.__doc__ = """transpose for axarray acts the same way than for numpy array but with axis labels

     However two keyword are different:

        add_unknown (Optional[bool]): if True add a dimension 1 to the array of a unknown 
            label
        squeeze_missing (Optional[bool]): drop axis if label is missing and dimension
            is 1 else raise ValueError    

    Example:
        >>> a = axarray( ones((10,4,5)), ["time", "y", "x"])
        >>> a.transpose( ["x", "y", "time"])        
            

     numpy doc of transpose is copied below 
    --------------------------------------
    """+np.transpose.__doc__

    reshape =  _decore_loose_axes(np.reshape)
    ravel   =  _decore_loose_axes(np.ravel)
    flatten =  _decore_loose_axes(np.ndarray.flatten)

    def _reduce_axis(self, freduce, axis=None, **kwargs):
        # Loop over axis
        if isinstance(axis,list):
            for ax in self._get_axes(axis):
                #if not isinstance(self,axarray): return self
                # initial is None after the first iteration
                self = self._reduce_axis(freduce, axis=ax, **kwargs)
            return self
        ##########################################################



        if axis is None:
            if kwargs.get("keepdims",False):
                return self.__class__(freduce( asarray(self), axis = None, **kwargs ), self.axes)
            else:
                return freduce(asarray(self), axis = None, **kwargs )

        iaxis = self.axis_index(axis)
        ndata = freduce(asarray(self), axis = iaxis, **kwargs )
        axes = list(self.axes)
        if len(ndata.shape) < len(self.shape):
            axes.remove( axis )
        elif len(ndata.shape) > len(self.shape):
            raise Exception("The freduce function cannot add dimension to the data")
        if not len(ndata.shape): return ndata
        return self.__class__(ndata, axes)

    def _reduce_func(self, freduce, axis=None,  initial=None):
        #########################################################
        # Loop over axis
        if isinstance(axis,list):
            for ax in self._get_axes(axis):
                if not isaxarray(self): return self
                # initial is None after the first iteration
                self = self._reduce_func(freduce, axis=ax, initial= (None if axis.index(ax) else initial) )
            return self
        ##########################################################

        if axis is None:
            if initial is None:
                return reduce(freduce, self.flat)
            else:
                return reduce(freduce, self.flat, initial)
        axes = list(self.axes)
        if initial is None:
            return reduce (freduce, self.transpose( [axis,None] ))
        else:
            return reduce (freduce, self.transpose( [axis,None] ), initial)

    def reduce(self, freduce, axis=None, initial=None):
        """ reduce the data along axis name(s) with the freduce func

        If the freduce method as signature f(A, axis=) (e.g. mean, std, max, etc...)
            f is called with its axis name as keyword argument
        If the freduce method as signature f(A), transpose the array so the given axis label
            is first and then call f(A)

        If axis is iterable freduce is executed for each axis items on the array resulting of the 
        precedent call. 

        Args:
            freduce : function to apply
            axis : axis label or list of label on witch the freduce function will be called. 
                if None (default) the array is flatten before executing freduce
            initial (Optional) : only used if freduce has signature f(A), that the itinital object 
                of the python reduce function.      

        """
        #########################################################
        # loop over axis
        if isinstance(axis, list):
            if initial is None:
                try:
                ##
                # To avoid confusion try first on the first axes
                # if succed do the rest
                    tmp = self._reduce_axis(freduce, axis=axis[0])
                    self = tmp
                    return self._reduce_axis(freduce, axis=axis[1:])
                except TypeError as e:
                    if "'axis'" in e.message:
                        return self._reduce_func(freduce, axis=axis)
                    else:
                        raise e
            else:
                return self._reduce_func(freduce, axis=axis, initial=initial)
        ##########################################################

        if initial is None:
            try:
                return self._reduce_axis(freduce, axis=axis)
            except TypeError as e:
                if "'axis'" in e.message:
                    return self._reduce_func(freduce, axis=axis)
                else:
                    raise e    
        return self._reduce_func(freduce, axis=axis, initial=initial)

    mean = _decore_reduce_func(np.mean)
    var  = _decore_reduce_func(np.var)
    std  = _decore_reduce_func(np.std)
    min  = _decore_reduce_func(np.min)
    max  = _decore_reduce_func(np.max)
    sum  = _decore_reduce_func(np.sum)
    prod = _decore_reduce_func(np.prod)

    argmax = _decore_reduce_func(np.argmax)
    argmin = _decore_reduce_func(np.argmin)

    cumsum =  _decore_reduce_func(np.cumsum)
    cumprod = _decore_reduce_func(np.cumprod)

    def squeeze(self, axis=None):
        """remove single-dimensional entries from the shape of an array.

        Args:
            a : array_like
                Input data.
            axis : None or axis labels 

                Selects a subset of the single-dimensional entries in the
                shape. If an axis is selected with shape entry greater than
                one, an error is raised.

        Returns:
            squeezed : axarray
            
        """
        if axis is None:
            shape = self.shape
            axes = [ax for i,ax in enumerate(self.axes) if shape[i]>1]

        elif hasattr(axis, "__iter__"):
            axes = [ax for ax in self.axes if ax not in axis]
        else:
            axes = [ax for ax in self.axes if ax != axis]

        return axarray( self.A.squeeze(), axes)

    @__lop__
    def __add__(x, y):
        return x+y
    @__lop__
    def __sub__(x, y):
        return x-y
    @__lop__
    def __mul__(x, y):
        return x*y
    @__lop__
    def __floordiv__(x, y):
        return x//y
    @__lop__
    def __mod__(x, y):
        return x%y
    @__lop__
    def __divmod__(x, y):
        return divmod(x,y)
    @__lop__
    def __pow__(x, y ):
        return pow(x,y)
    @__lop__
    def __lshift__(x, y):
        return x<<y
    @__lop__
    def __rshift__(x, y):
        return x>>y
    @__lop__
    def __and__(x, y):
        return x&y
    @__lop__
    def __xor__(x, y):
        return x^y
    @__lop__
    def __or__(x, y):
        return x|y

    @__rop__
    def __radd__(x, y):
        return x+y
    @__rop__
    def __rsub__(x, y):
        return x-y
    @__rop__
    def __rmul__(x, y):
        return x*y
    @__rop__
    def __rdiv__(x, y):
        return x/y
    @__rop__
    def __rtruediv__(x, y):
        return x/y
    @__rop__
    def __rfloordiv__(x, y):
        return x/y
    @__rop__
    def __rmod__(x, y):
        return x%y
    @__rop__
    def __rdivmod__(x, y):
        return divmod(x,y)
    @__rop__
    def __rpow__(x, y ):
        return pow(x,y)
    @__rop__
    def __rlshift__(x, y):
        return x<<y
    @__rop__
    def __rrshift__(x, y):
        return x>>y
    @__rop__
    def __rand__(x, y):
        return x&y
    @__rop__
    def __rxor__(x, y):
        return x^y
    @__rop__
    def __ror__(x, y):
        return x|y

    @__uop__
    def __neg__(x):
        return -x
    @__uop__
    def __pos__(x):
        return +x
    @__uop__
    def __abs__(x):
        return abs(x)
    @__uop__
    def __invert__(x):
        return ~x
    @__uop__
    def __complex__(x):
        return complex(x)
    @__uop__
    def __int__(x):
        return int(x)
    @__uop__
    def __long__(x):
        return long(x)
    @__uop__
    def __float__(x):
        return float(x)
    @__uop__
    def __index__(x):
        return x.__index__()

