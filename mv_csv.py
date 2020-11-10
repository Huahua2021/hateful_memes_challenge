import pandas as pd
import numpy as np
import os
import shutil

if not os.path.exists("csv"):
    os.mkdir("csv")

files = os.listdir("./save")
for file in files:
    if len(file) > 13 and file[:13] == "hateful_memes":
        fold = file

reports = os.path.join(os.getcwd(), "save", fold, "reports")

files = os.listdir(reports)
for file in files:
    if len(file) > 13 and file[:13] == "hateful_memes":
        csv = file

absolute_csv = os.path.join(reports, csv)

print(absolute_csv)
shutil.move(absolute_csv, "./csv")
shutil.rmtree("save")
