# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 

@author: Steve
"""

import os

def create_file(new_directory_name):
    # Check if the directory already exists
    if not os.path.exists(new_directory_name):
        # Create the new directory
        os.mkdir(new_directory_name)
        print(f"Directory '{new_directory_name}' created successfully.")
    else:
        print(f"Directory '{new_directory_name}' already exists.")
        
