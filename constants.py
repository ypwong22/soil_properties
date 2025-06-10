import os
import numpy as np

path_data = os.path.join(
    os.environ['SHARDIR'], 'Soil_Properties', 'data'
)

path_intrim = os.path.join(
    os.environ['SHARDIR'], 'Soil_Properties', 'intermediate'
)

elm_interface = np.array([0, 1.75, 4.51, 9.06, 16.55, 28.91, 49.29, 82.89, 138.28, 229.61, 380.19])
elm_nodes = [0.71, 2.79, 6.23, 11.89, 21.22, 36.61, 61.98, 103.8, 172.8, 286.46]
elm_thickness = np.diff(elm_interface)
