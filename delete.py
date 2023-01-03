import pandas as pd

obj = pd.read_pickle(r'object_dictionary.pkl')
print(obj)

for o in obj.values():
    print(o.x)
