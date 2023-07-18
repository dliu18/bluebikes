import pandas as pd
import numpy as np

def largest_category(df, key, categories, prefix=""):
    values = []
    for category in categories:
        values.append(df.loc[key][prefix + category])
    return categories[np.argmax(values)]