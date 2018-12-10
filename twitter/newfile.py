from os import walk
import pandas as pd
import os

filenames=[]
for(dirpath, dirnames,files) in walk("data/clean"):
    filenames.extend(files)
    break
os.chdir("data/clean/")
combined_csv = pd.concat([pd.read_csv(f) for f in filenames])
combined_csv.to_csv("../../data/combined_csv5.csv", index=False)
print(len(combined_csv))

