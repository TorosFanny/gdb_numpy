# -*- coding: utf-8 -*-
"""
For use in the gdb debugger to map C/C++ vector and array variables 
into numpy array for visualization or manipulations. For example, a 
vector variable can be mapped into a numpy array and plotted using 
matplotlib during a debugging session.  
"""



import gdb
import numpy as np
import itertools
import functools
import deref

#Maps to C/C++ numeric types into the corresponding data type in 
#numpy
_type_list = {'unsigned char': np.uint8, 
              'unsigned int': np.uint32,
              'char': np.int8, 
              'short': np.int16,
              'int': np.int32, 
              'float': np.float32,
              'double': np.float64}

#A regular expression used for checking the data type in C/C++
_type_exp = '('+functools.reduce(lambda x,y: x+'|'+y,_type_list.keys())+')'

#The list of classes that provides the utilities to dereference 
#pointer types. These are defined in the module 'deref'.
_ptr_list = [deref.DeRefPtr, deref.DeRefArr]
#The lis of classes that provides utilities to dereference container 
#types. These are defined in the module 'deref'.
_container_list = [deref.DeRefVec]


def to_array(var,shape = None):
    """
    Converts C/C++ vector/array into numpy array. If some dimensions 
    are pointers, then the shape parameter must be supplied to 
    provide bounds for these pointers. 
    
    Parameters
    ----------
    var: string 
        Indicates the name of the C/C++ variable.

    shape: tuple of int
        This is only used and needed when the variable contains 
        pointers. Has no effect if the type is not a pointer. 
        See examples.

    Examples
    -------- 
    If mat is a C/C++ variable:
        
    float mat[10][5] = ...;

    Then the following in gdb will give a numpy array:

    > py mat = gdb_numpy.to_array('mat')

    > py print mat.shape

    (10,5)
            
    The shape parameter has no effect for array or vector types: 
    
    > py mat = gdb_numpy.to_array('mat',(1,2))
    
    > py print mat.shape
    
    (10,5)

    If mat is a pointer type, however, then the shape parameter 
    will be used to construct a numpy array of appropriate shape. 
    
    float** mat = ...;
    
    >py mat = gdb_numpy.to_array('mat',(10,5))

    >py print mat.shape

    (10,5)
    
    Error will arise if shape parameter is not provided. 
    
    >py mat = gdb_numpy.to_array('mat')

    IndexError: Insufficient number of bounds. 
    A bound is needed for each dimension 
    corresponding to a * or []."
    """
    val = gdb.parse_and_eval(var)
    deref_func, arg_no, bounds, dtype = _get_deref_funcs(val, shape)
    ranges = [range(bound) for bound in bounds]
    indices = itertools.product(*ranges)
    narray = np.zeros(bounds, dtype = _type_list[dtype])
    for index in indices:
        start_ind = 0; val = gdb.parse_and_eval(var)
        for func, arg_n in zip(deref_func,arg_no):
            end_ind = start_ind + arg_n
            val = func(val,index[start_ind:end_ind])
            start_ind = end_ind
        narray[index] = _type_list[dtype](val)    
    return narray

def _get_deref_funcs(val,shape = None):
    #Used by to_array to extract the underlying data in a 
    #gdb.Value type that corresponds to a vector or array.
    #Given a gdb.Value type, returns a list of functions 
    #that are needed to dereference the type. 
    arr_type = _get_type(val)
    old_type = None
    shape_ind = None
    deref_func = []
    arg_no = []
    bounds = []
    if shape:
        shape_ind = 0
    while(arr_type != old_type):
        while(arr_type != old_type):
            arr_type = _get_type(val)
            old_type = arr_type
            val,shape_ind = _deref(_ptr_list, val,deref_func,
                                   bounds, arg_no, shape_ind, shape)
        val, shape_ind = _deref(_container_list, val, deref_func,
                                bounds, arg_no, shape_ind, shape)
        arr_type = _get_type(val)
    if shape_ind:
        if shape_ind != len(shape):
            print("Not all of the bounds are used.")
    return deref_func, arg_no, bounds, arr_type
                        
                        
def _deref(type_list,val,deref_func,bounds,arg_no,shape_ind,shape):
    #Dereference val and return the dereferenced type. 
    arr_type = _get_type(val)
    for ref_type in type_list:
        if(ref_type.pattern.search(arr_type)):
            deref_ptr = ref_type(val,shape_ind,shape)
            val = deref_ptr.val
            arr_type = _get_type(val)
            shape_ind = deref_ptr.shape_ind
            deref_func.append(deref_ptr.deref)
            bounds.extend(deref_ptr.bounds)
            arg_no.append(len(deref_ptr.bounds))
    return val,shape_ind

def _get_type(val):
    #Returns the type name with qualifiers and typedefs removed. 
    return str(val.type.strip_typedefs().unqualified())
