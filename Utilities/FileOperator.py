import pickle
from telnetlib import OLD_ENVIRON
import uuid
import os
import datetime

def save_obj(obj, name):
    """
    Saves given object as a pickle file
    :param obj:
    :param name:
    :return:
    """
    if name[-4:] != ".pkl":
        name += ".pkl"
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """
    Loads a pickle file object
    :param name:
    :return:
    """
    with open(name, 'rb') as f:
        return pickle.load(f)


def init_file_name():
    dayTime = datetime.datetime.now().strftime('%Y-%m-%d')
    hourTime = datetime.datetime.now().strftime('%H-%M-%S')
    pwd = "/home/neardws/Documents/Game-Theoretic-Deep-Reinforcement-Learning/Data/" + dayTime + '-' + hourTime

    if not os.path.exists(pwd):
        os.makedirs(pwd)

    uuid_str = uuid.uuid4().hex
    convex_environment_name = pwd + '/' + 'convex_environment_%s.pkl' % uuid_str
    random_environment_name = pwd + '/' + 'random_environment_%s.pkl' % uuid_str
    local_environment_name = pwd + '/' + 'local_environment_%s.pkl' % uuid_str
    edge_environment_name = pwd + '/' + 'edge_environment_%s.pkl' % uuid_str
    old_environment_name = pwd + '/' + 'old_environment_%s.pkl' % uuid_str
    global_environment_name = pwd + '/' + 'global_environment_%s.pkl' % uuid_str
    
    return {
        "convex_environment_name": convex_environment_name,
        "random_environment_name": random_environment_name,
        "local_environment_name": local_environment_name,
        "edge_environment_name": edge_environment_name,
        "old_environment_name": old_environment_name,
        "global_environment_name": global_environment_name,
    }