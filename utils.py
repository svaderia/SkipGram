#!/usr/bin/env python
# @author = 53 68 79 61 6D 61 6C 
# date	  = 20/12/2017

import os

def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
    	pass
