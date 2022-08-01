import pymongo as pm

def grab_results(dbname, collname, exp_id, 
                 port=27017, idx=-1):
    conn = pm.MongoClient(port=port)
    coll = conn[dbname][collname]
    query = {'exp_id': exp_id}
    cursor = coll.find(query)
    result_list = [rec for rec in cursor]
    if idx is None:
        return result_list
    return result_list[idx]