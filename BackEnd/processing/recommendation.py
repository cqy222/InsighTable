import pandas as pd

class harpoonProcess:
    def __init__(self, path, row, col):
        self.path = path
        self.header_row = row
        self.header_col = col

    def process(self):
        df = pd.read_excel(self.path, index_col=self.header_col, header=self.header_row)

        # 调用rl，获得变换结果的pandas 这里采用最简单的一个pandas操作代替
        # df = df.swaplevel(axis=1, i=0, j=1).sort_index(axis=1)

        print(df)

        return df
