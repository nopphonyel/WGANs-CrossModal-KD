import os
import sys


def get_weights_collection_path(sub_dir=None):
    try:
        p = os.environ['PYTHONPATH'].split(os.pathsep)[0]
        if sub_dir is not None:
            p = os.path.join(p, sub_dir)
        return os.path.join(p, "weig_coll")
    except KeyError:
        return ""
