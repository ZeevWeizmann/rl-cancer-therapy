import numpy as np

# tumor growth
ALPHA_S = 0.03
ALPHA_R = 0.025
BETA = 0.045
K = 1e7

# competition
C_SR = 1.0
C_RS = 2.0

# simulation
DT = 1.0
MAX_STEPS = 400

# treatment
DOSES = np.array([0.0, 0.5, 1.0])
DOSE_PENALTY = 0.2   

# scaling
SCALE = 1e7

# initial tumor
TS0 = 1e6
TR0 = 1e5