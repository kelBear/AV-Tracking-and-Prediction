from bayesian_model import *
import json

def intersection(c, p, a, s = True):
    network = crossintersection()

    observations = {}
    if c != -1:
        car = str(bool(c))
        observations['car'] = car
    if p != -1:
        ped = str(bool(p))
        observations['ped'] = ped
    if a != -1:
        approach = str(bool(a))
        observations['approach'] = approach
    stop = str(bool(s))
    observations['stop'] = stop

    beliefs = map(str, network.predict_proba(observations))
    joutput = json.loads(beliefs[4])
    return joutput['parameters'][0]['True']