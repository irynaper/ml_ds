from pycaret.datasets import get_data
from pycaret.classification import *

data = get_data('glass')
print(data.head())

exp = setup(data=data, target='Type', session_id=123)
best_model = compare_models()

save_model(best_model, 'final_glass_model')