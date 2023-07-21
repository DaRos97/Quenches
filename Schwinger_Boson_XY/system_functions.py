import numpy as np
from pathlib import Path
import inputs as inp
import csv
import os
#Save the dictionaries in the file given, rewriting the already existing data if precision is better

def SaveToCsv(Data,csvfile):
    header = Data.keys()
    with open(csvfile,'a') as f:
        writer = csv.DictWriter(f, fieldnames = header)
        writer.writeheader()
        writer.writerow(Data)




