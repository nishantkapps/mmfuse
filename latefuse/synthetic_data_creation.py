import csv
import random
import pandas as pd

with open("synthetic_data.csv", "a", newline='') as f:
    writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    index = 6001

    for i in range(100):
        X = 0
        Y = 0
        t = 0
        for _ in range(60):
            Xold = X
            Yold = Y
            Y = random.random()/10
            X += 0.05 + ((random.random()/10) - 0.05) 
            t += 0.34
            index += 1
            writer.writerow([index , t, X, Y, X - Xold, Y - Yold, "right", 1.0])

df = pd.read_csv("synethic_data.csv")
df.set_index('id', inplace = True)
df.drop()