from flask import Flask, request, render_template
import joblib
import pandas as pd
import os
from ultralytics import YOLO
model = joblib.load("pcos_model.pkl")
print(model.feature_names_in_)

# import pandas as pd
# df = pd.read_csv(r"C:\Users\Elitebook 840 G6\Documents\AVAL2\data\Final_PCOS_Dataset.csv")
# print(df.columns)