import numpy as np


V_INDICES = {"V1": 6, "V2": 7, "V3": 8, "V4": 9, "V5": 10, "V6": 11}

COMBINATIONS = {
    "V2": [("V1", "V4"), ("V1", "V3")],
    "V3": [("V2", "V4"), ("V1", "V5")],
    "V4": [("V3", "V5"), ("V2", "V5"), ("V1", "V6")],
    "V5": [("V4", "V6"), ("V3", "V6")],
}


def simvea_h_augment(ecg, p=0.3, alpha_range=(0.1, 1.0), beta_range=(0.0, 2.0)):
    out = ecg.copy()
    src = ecg
    for lead in ("V2", "V3", "V4", "V5"):
        if np.random.rand() > p:
            continue
        combos = COMBINATIONS[lead]
        j_name, z_name = combos[np.random.randint(len(combos))]
        i, j, z = V_INDICES[lead], V_INDICES[j_name], V_INDICES[z_name]
        alpha = np.random.uniform(*alpha_range)
        beta = np.random.uniform(*beta_range)
        out[i] = alpha * (beta * src[j] + (2.0 - beta) * src[z])
    return out
