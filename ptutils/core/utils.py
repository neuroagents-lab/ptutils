import pymongo as pm
import os
import random
import numpy as np
import torch

def grab_results(dbname, collname, exp_id, port=27017, idx=-1):
    conn = pm.MongoClient(port=port)
    coll = conn[dbname][collname]
    query = {"exp_id": exp_id}
    cursor = coll.find(query)
    result_list = [rec for rec in cursor]
    if idx is None:
        return result_list
    return result_list[idx]

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multi-GPU
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
