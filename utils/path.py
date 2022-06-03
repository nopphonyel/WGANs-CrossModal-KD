import os
import sys


def get_weights_collection_path():
    try:
        p = os.environ['PYTHONPATH'].split(os.pathsep)[0]
        return os.path.join(p, "weig_coll")
    except KeyError:
        return ""
