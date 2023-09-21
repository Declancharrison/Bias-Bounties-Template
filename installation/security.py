import sys
import builtins
import dill as pkl
import io
import builtins
from contextlib import contextmanager
from io import StringIO
import dis

import pandas as pd 
import numpy
import scipy
from sklearn import *
import xgboost
import lightgbm
import collections

x_train = pd.read_csv("/home/non-root/training_data.csv", index_col=False)

safe_builtins = [
    'range',
    'complex',
    'set',
    'frozenset',
    'slice',
    '_load_type',
    'getattr',
    'setattr',
    '__dict__',
    '__main__'
]

class RestrictedUnpickler(pkl.Unpickler):

    def find_class(self, module, name):
        if 'numpy' in module:
            return getattr(sys.modules[module], name)
        if 'scipy' in module:
            return getattr(sys.modules[module], name)
        if 'sklearn' in module or 'xgboost' in module or 'lightgbm' in module or 'collections' in module:
            if 'predict' in name:
                return getattr(sys.modules[module], name.split('.')[0]).predict
            if 'transform' in name:
                return getattr(sys.modules[module], name.split('.')[0]).transform
            return getattr(sys.modules[module], name)
        # Only allow safe classes from builtins.
        if 'dill' in module:
            return getattr(sys.modules[module], name)
        if ("__builtin__" in module or "builtins" in module) and name in safe_builtins:
            if name == '__main__':
                return
            return getattr(builtins, name)
        # Forbid everything else.
        raise pkl.UnpicklingError("global '%s.%s' is forbidden" %
                                     (module, name))

def restricted_loads(s):
    """Helper function analogous to pickle.loads()."""
    return RestrictedUnpickler(io.BytesIO(s)).load()

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
@contextmanager
def captureStdOut(output):
    stdout = sys.stdout
    sys.stdout = output
    try:
        yield
    finally:
        sys.stdout = stdout
        
def get_instructions(func):
    out = StringIO()
    with captureStdOut(out):
        dis.dis(func)

    return [AttrDict({
               'opname': i[16:36].strip(),
               'arg': i[37:42].strip() or 0,
               'argval': i[44:-1].strip()
           }) for i in out.getvalue().split("\n")]


def security_call_1(filepath, type_model):
    try:
        with open(filepath, "rb") as file:
            item_data = file.read()
        item = restricted_loads(item_data)
    except:
        comment = f"Model ({type_model}) could not be properly loaded with pickle, please see instructions for downloading and submitting pairs!\n"
        return comment, None
    return "", item

def security_call_2(item, type_model):
    try:
        
        with open(f'bad_argvals.txt', 'r') as file:
            bad_argvals = [line.strip() for line in file]
            
        if 'sklearn' in str(item.__module__) or 'xgboost' in str(item.__module__):
            return ""
        instructions = get_instructions(item)
        for instruction in instructions:
            argval = set(instruction['argval'].replace("'","").split(' '))
            if ((instruction['opname'] == "IMPORT_NAME") or (len(argval.intersection(set(bad_argvals))) != 0)):
                comment = f"Potential bad ({type_model})model!\n"
                return comment
    except:
        comment = f"Error in disassembling ({type_model})model!\n"
        return comment
    return ""

def sizecheck(obj, name):
    # get group indices
    try:
        indices = obj(x_train)
        obj_check = indices.shape == (len(x_train),)
    except:
        comment = f"Problem in {name} function processing inputs!\n"
        return comment

    if obj_check == False:
        comment = f"Problem in {name} function dimensions!\n"
        return comment

    if name == "group":
        if indices.sum() == 0:
            comment = f"Empty Group Function!\n"
            return comment
    return ""

def main():
    # security call 1
    comment_group, group = security_call_1(f"container_tmp/group.pkl", "group")
    comment_hypothesis, hypothesis = security_call_1(f"container_tmp/hypothesis.pkl", "hypothesis")

    if comment_group or comment_hypothesis:
        return comment_group + comment_hypothesis
    
    # security call 2
    comment_group = security_call_2(group, "group")
    comment_hypothesis = security_call_2(hypothesis, "hypothesis")

    if comment_group or comment_hypothesis:
        return comment_group + comment_hypothesis
    
    # check input/output dimensions
    comment_group = sizecheck(group, 'group')
    comment_hypothesis = sizecheck(hypothesis, 'hypothesis')

    if comment_group or comment_hypothesis:
        return comment_group + comment_hypothesis

    return "safe"

if __name__ == "__main__":
    comment = main()
    exit(comment)