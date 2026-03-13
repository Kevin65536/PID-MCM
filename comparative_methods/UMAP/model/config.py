import itertools
import random
import numpy as np
from pathlib import Path
from addict import Dict
from copy import deepcopy
import yaml

import math



class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

# class Config(Dict, metaclass=Singleton):


def get_param_sets(config:Dict, *, dosage:str='greed', sample_n:int=1) -> list:
    """将config中列表形式的超参数进行排列，形成超参数组合。

    :return list: list中的元素为字典，一个字典代表一组超参数(该字典的键与self.config中的键相同)
    """
    cfs = []
    iter_paras_name = []
    iter_paras_value = []
    cf = Dict()
    for key, value in config.items():
        if isinstance(value, Dict) and 'flag' in value.keys():
            assert value.flag in ['arange', 'linspace']

            if value.flag == 'arange':
                iter_paras_name.append(key)
                iter_paras_value.append(list(np.arange(value.start, value.stop, value.step)))
            if value.flag == 'linspace':
                iter_paras_name.append(key)
                iter_paras_value.append(list(np.linspace(value.start, value.stop, value.number)))

        if isinstance(value, list):
            iter_paras_name.append(key)
            iter_paras_value.append(value)
        else:
            cf[key] = value
    try:
        assert len(iter_paras_name) == len(iter_paras_value)
    except:
        print('Something wrong with the iterable para in config')
        # TODO logger

    iter_paras = list(itertools.product(*iter_paras_value, repeat=1))
    if dosage == 'random':
        iter_paras = random.sample(iter_paras, sample_n)
    for iter_para in iter_paras:
        for name, value in zip(iter_paras_name, iter_para):
            cf[name] = value
        cfs.append(deepcopy(cf)) 
    return cfs

class Config(Dict):
    '''
    配置类。继承自Dict，以便于使用。
    '''
    def __init__(self, config_path:Path):
        '''
        给定配置文件路径，生成配置实例。

        :param Path config_path: yaml文件路径。
        '''
        super().__init__()
        self._load_config(config_path)

    def append(self, config_path:Path):
        '''
        追加配置内容，例如可通过独立文件增配模型训练参数。

        Note: 若追加配置中的某些key与已有配置key名称相同将覆盖原有配置的key内容。

        :param Path config_path: yaml文件路径。
        '''
        self._load_config(config_path)

    def _load_config(self, config: Path):
        with open(config) as f:
            settings = Dict(yaml.load(f, Loader=yaml.FullLoader))
        self.update(settings)

    def save_to_yaml(self, data, save_path):
        """
        存储yaml文件
        """
        with open(save_path, "w") as f:
            yaml.dump(data, f)


