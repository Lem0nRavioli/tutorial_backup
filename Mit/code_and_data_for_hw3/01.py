import numpy as np
from code_for_hw3_part2 import load_auto_data


data = load_auto_data('auto-mpg.tsv')

print(type(data))
print(np.shape(data))
print(data)