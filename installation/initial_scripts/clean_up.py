#!/usr/bin/python3
import sys
import os

def clean_up_branch(branch_name):
    os.system(f"git push origin -d {branch_name}")
    return

clean_up_branch(sys.argv[1])