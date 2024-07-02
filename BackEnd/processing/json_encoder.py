import numpy as np
import json

class NpEncoder(json.JSONEncoder):      # 解决json.dumps对于numpy数据类型的问题
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        else:
            return super(NpEncoder, self).default(obj)