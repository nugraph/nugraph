import pandas as pd
import particle

class StoppingMuon :
    def __init__(self):
        self._labels = ['signal', 'background']
		
    @property
    def labels(self):
        return self._labels
		
    def label(self, idx: int):
        if not 0 <= label < len(self._labels):
            raise Exception(f'index {idx} out of range for {len(self._labels)} labels.')
        return self._labels[idx]

    def index(self, name: str):
        if name not in self._labels:
            raise Exception(f'"{name}" is not the name of a class.')
        return self._labels.index(name)
        
    @property
    def signal(self):
        return self.index('signal')
        
    @property
    def background(self):
        return self.index('background')
        
    def __call__(self,  part: pd.DataFrame):
        print('STOPPING MUON filter')
        #print("part type:", type(part))
        #print("part columns:", getattr(part, 'columns', 'No columns attribute'))
        #print("part head:\n", getattr(part, 'head', lambda: part)())
        
        #Show what are inside the columns
        #print(part.columns.tolist())

        #Get all primary particles
        #part = part.set_index("g4_id", drop=False)

        primaries = part[(part.parent_id==0)]

        #print('x_postion = ',x)
        # Check particle type (or PDG) for each primaries
        for _, primary in primaries.iterrows():
            if primary.type in (-13,13) and -34 < primary.end_position_x < 34 and -0.2 <primary.end_position_y < 102 and -3.9 < primary.end_position_z < 120:
                print("✅ ***** Signal ***** ✅")
                print('primary', primary.type)
                print("primary.end_position_x = ", primary.end_position_x)
                print("primary.end_position_y = ", primary.end_position_y)
                print("primary.end_position_z = ", primary.end_position_z)
                return self.signal
        
        print("❌ ***** Background ***** ❌")
        print('primary', primary.type)
        #print("primary.end_position_x = ", primary.end_position_x)
        #print("primary.end_position_y = ", primary.end_position_y)
        #print("primary.end_position_z = ", primary.end_position_z)
        return self.background
        
