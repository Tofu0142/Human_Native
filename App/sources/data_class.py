from typing import List, Dict, Any, Optional
import re
import pandas as pd
import numpy as np

# data model
class Dataset:
    def __init__(self, org_id: str, id: str, name: str, type: str):
        self.org_id = org_id
        self.id = id
        self.name = name
        self.type = type

class Data:
    def __init__(self, dataset_id: str, id: str, value: str, flag: bool = False):
        self.dataset_id = dataset_id
        self.id = id
        self.value = value
        self.flag = flag

