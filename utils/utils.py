#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 11:53:22 2024

@author: daltonm
"""

def convert_string_inputs_to_int_float_or_bool(orig_var):
    if type(orig_var) == str:
        orig_var = [orig_var]
    
    converted_var = []
    for v in orig_var:
        v = v.lower()
        try:
            v = int(v)
        except:
            pass
        try:
            v = float(v)
        except:
            v = None  if v == 'none'  else v
            v = True  if v == 'true'  else v
            v = False if v == 'false' else v
        converted_var.append(v)
    
    if len(converted_var) == 1:
        converted_var = converted_var[0]
            
    return converted_var