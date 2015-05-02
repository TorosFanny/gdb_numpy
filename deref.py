# -*- coding: utf-8 -*-
"""
For use by the gdb_numpy module. Provides utilities to dereference 
gdb.Value types that corresponds to vectors/arrays in C/C++
"""
import abc
import re
import gdb

class DeRefBase(object):
    """
    Abstract base class that provides functionalities for 
    dereferencing. Upon initialization, it will provide information 
    about how to dereference the input type, together with the shape. 
    
    Parameters
    ----------
    
    val: gdb.Value
        A gdb.Value type that is created from a C/C++ vector/array. 
    
    shape_ind: int 
        A parameter used for bookkeeping. 
        
    shape: tuple
        Parameter used for providing the shape of the array. 
    """
    __metaclass = abc.ABCMeta    
    
    def __init__(self,val,shape_ind,shape):
        super(DeRefBase,self).__init__()
        self.bounds=[]
        self.val = val
        self.shape_ind = shape_ind
        self._shape = shape
    
    @abc.abstractmethod
    def deref(self,val,indices):
        """
        Custom method for dereferencing. 
        
        Parameters
        ----------
        
        val: gdb.Value
            Corresponding to the data type that is to be 
            dereferenced. 
            
        indices: tuple
            Indices use for dereferencing. 
        """
        pass
                
    def _get_range_from_shape(self,arg_no):
        #Obtain shape information for the array/vector
        #from the parameter shape.
        #arg_no is the number of indices used in the method 
        #deref and shape is a use provide parameter that 
        #specifies the length of the vector.         
        shape_ind = self.shape_ind
        shape = self._shape
        end_ind = shape_ind + arg_no
        if end_ind > len(shape): 
            raise IndexError("Insufficient number of bounds. "
                             "A bound is needed for each dimension "
                             "corresponding to a * or [].")
        self.bounds = shape[shape_ind:end_ind]
        self.shape_ind = end_ind
        
    
class DeRefPtr(DeRefBase):
    """
    Provides functionalities for dereferencing pointer types. 
    Used by gdb_numpy.
    """
    
    #Regular expressions check for pointer types in a 
    #gdb.Type variable.
    pattern = re.compile('\*$|\*\)(?:\[\d*\])+$')

    def __init__(self,val,shape_ind,shape):
        super(DeRefPtr,self).__init__(val,shape_ind,shape)
        if shape is None:
            raise ValueError("Array shape must be specified "
                             "for pointer type.")
        self._arg_no = 1
        self._update(self._shape)

    def deref(self,val,indices):
        """
        Use for dereferencing pointer. Note that indices should be 
        a tuple or list of the form [index].
        """
        return val[indices[0]] 
        
    def _update(self,shape):
        self.val = self.deref(self.val,[0])
        self._get_range_from_shape(self._arg_no)
    
class DeRefArr(DeRefBase):
    """
    Provides functionalities for dereferencing array types. 
    """
    
    #Regular expressions check for array types in a 
    #gdb.Type variable.
    pattern = re.compile('(?:\[\d*\])+$')

    def __init__(self,val,shape_ind,shape):
        super(DeRefArr,self).__init__(val,shape_ind,shape)
        self._arg_no = 1
        self._update()
        
    def deref(self,val,indices):
        """
        Use for dereferencing array. Note that indices should be 
        a tuple or list of the form [index].
        """
        return val[indices[0]]
        
    def _update(self):
        val = self.val
        arr_type = str(val.type.unqualified().strip_typedefs())
        if re.search('(?:\[\])+',arr_type):
            raise gdb.error("Missing array dimensions. "
                            "All array dimensions must be "
                            "declared as const. ")
        match = self.pattern.search(arr_type)
        bounds = match.group()[1:-1]
        bounds = bounds.split('][')
        bounds = bounds[0]
        self.bounds = [int(bounds)]
        self.val = self.deref(val,[0])
        
class DeRefVec(DeRefBase):
    """
    Provides functionalities for dereferencing vector types. 
    """
    
    #Regular expressions check for vector types in a 
    #gdb.Type variable.
    pattern = re.compile('^(std::vector)')

    def __init__(self,val,shape_ind,shape):
        super(DeRefVec,self).__init__(val,shape_ind,shape)
        self._update()

    def deref(self,val,indices):
        """
        Use for dereferencing vector. Note that indices should be 
        a tuple or list of the form [index].
        """
        return val['_M_impl']['_M_start'][indices[0]]
        
    def _update(self):
        val = self.val
        self.val = self.deref(val,[0])
        v_start = val['_M_impl']['_M_start']
        v_end = val['_M_impl']['_M_finish']
        self.bounds = [v_end - v_start]