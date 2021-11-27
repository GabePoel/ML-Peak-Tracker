import os
import importlib
from os import path
from npdoc_to_md import render_md_from_obj_docstring

here = os.getcwd()

def import_from(module, name):
    module = __import__(module, fromlist=[name])
    return getattr(module, name)

def create_doc(module, module_name=None):
    # if module_name is None:
    #     module_name = 'lf.module'
    # parent = 'peak_finder.' + module.split('.')[0]
    # name = module.split('.')[-1]
    # print(parent)
    # print(name)
    # obj = import_from(parent, name)
    # obj = importlib.import_module('peak_finder.' + module)
    md = render_md_from_obj_docstring(obj=module, obj_namespace=module_name, remove_doctest_blanklines=False)
    fp = path.join(here, module_name).replace('.', '/') + '.md'
    os.makedirs(path.dirname(fp), exist_ok=True)
    f = open(fp, 'w')
    f.write(md)
    f.close()

modules = [
    'preview',
    'color_selection',
    'mistake_selection',
    'live_selection',
    'point_selection']

# for module in modules:
#     m = 'live_fitting.' + module
#     create_doc(m)

from peak_finder import live_fitting as lf

create_doc(lf.preview, 'live_fitting.preview')
create_doc(lf.color_selection, 'live_fitting.color_selection')
create_doc(lf.mistake_selection, 'live_fitting.mistake_selection')
create_doc(lf.live_selection, 'live_fitting.live_selection')
create_doc(lf.point_selection, 'live_fitting.point_selection')