#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os


# In[ ]:


LOCAL_FILE = "arrythmia.data"
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data"


# In[ ]:


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


# In[ ]:


source = LOCAL_FILE if os.path.exists(LOCAL_FILE) else URL
raw = pd.read_csv(source, header=None, na_values="?")
print(raw.shape)


# In[ ]:


first_cols = [
    "age", "sex", "height", "weight", 
    "qrs_duration", "pr_interval", "qt_interval", "t_interval", "p_interval",
    "qrs_angle", "t_angle", "p_angle", "qrst_angle", "j_angle", "heart_rate"]


# In[ ]:


df = raw.iloc[:, list(range(15)) + [279]].copy()
df.columns = first_cols + ["class"]


# In[ ]:


df["sex"] = df["sex"].map({0: "Male", 1: "Female"})


# In[ ]:


class_names = {
    1: "Normal", 2: "Ischemic changes (CAD)", 3: "Old anterior MI",
    4: "Old inferior MI", 5: "Sinus tachycardia", 6: "Sinus Bradycardia",
    7: "Ventricular premature contraction", 8: "Supraventricular premature contraction",
    9: "Left bundle branch block", 10: "Right bundle branch block",
    11: "1st degree AV block", 12: "2nd degree AV block", 13: "3rd degree AV block",
    14: "Left ventricular hypertrophy", 15: "Atrial fibrillation/flutter", 16: "Other",}



# In[ ]:


# A simple two-group label that is useful for comparison
df["diagnosis"] = df["class"].map(class_names)


# In[ ]:


# Age bands for grouping/pivoting
df["age_group"] = pd.cut(df["age"], bins=[0, 30, 45, 60, 120], 
                         labels=["<30", "30-44", "45-59", "60+"], right=False)


# In[ ]:


# Add a patient ID so rows can be tracked after reshpaing
df.insert(0, "patient_id", range(1, len(df) + 1))


# In[ ]:


print(df.head())
print("\nMissing values per column:\n", df.isna().sum()[df.isna().sum() > 0])


# In[ ]:


# 1) Melt
interval_cols = ["qrs_duration", "pr_interval", "qt_interval", "p_interval"]


# In[ ]:


df["status"] = np.where(df["class"] == 1, "Normal", "Arrhythmia")
print(df.columns.tolist())


# In[ ]:


long_df = pd.melt(
    df, 
    id_vars=["patient_id", "sex", "status"], value_vars=interval_cols, var_name="measurement", value_name="ms")


# In[ ]:


print(long_df.head(8))
print("\nWide shape:", df[["patient_id"] + interval_cols].shape, "-> Long shape:", long_df.shape)


# In[ ]:


#2) Pivot, average heart rate by diagnosis status and sex
# pivot: rebuild the wide table from the melted one
wide_again = long_df.pivot(index="patient_id", columns="measurement", values="ms")
print(wide_again.head())


# In[ ]:


original = df.set_index("patient_id")[interval_cols]
print("\nRound trip matches original:", wide_again[interval_cols].equals(original))


# In[ ]:


# pivot_table: Summarise with an aggregation function
hr_table = pd.pivot_table(df, index="age_group", columns="sex", values="heart_rate", aggfunc="mean", observed=True,).round(1)
print("Mean heart rate (bpm) by age group and sex:\n", hr_table)


# In[ ]:


# 3) Aggregation
print(df[["age", "weight", "heart_rate", "qrs_duration"]].agg(["mean", "median", "min", "max", "std"]).round(1))


# In[ ]:


summary = df.agg({"patient_id": "count", "age": "mean", "qt_interval": ["min", "max"], "heart_rate": lambda s: s.isna().sum()})
print(summary)


# In[ ]:


# 4) Iteration, flagging patients whose **QRS duration is 120 ms or longer**
# iterrows(): print the first five patients
for idx, row in df.head(5).iterrows():
    print(f"Patient {row['patient_id']}: {row['age']}y/o {row['sex']}, "f"HR={row['heart_rate']}, diagnosis={row['diagnosis']}")


# In[ ]:


#Itertuples(): count wide-QRS patients and record their diagnoses
wide_qrs = []
for r in df.itertuples(index=False):
    if r.qrs_duration >= 120:
        wide_qrs.append((r.patient_id, r.qrs_duration, r.diagnosis))


# In[ ]:


print(f"{len(wide_qrs)} patients have QRS duration >= 120 ms")
print(pd.DataFrame(wide_qrs, columns=["patient_id", "qrs_duration", "diagnosis"])["diagnosis"].value_counts())


# In[ ]:


# Iterating over columns with items(): report missing data
for col_name, col_values in df[first_cols].items(): 
    n_missing = col_values.isna().sum()
    if n_missing > 0:
        print(f"{col_name}: {n_missing} missing ({n_missing / len(df):.1%})")


# In[ ]:


# 5) Groupby
# How many patient per diagnosis?
print(df.groupby("diagnosis")["patient_id"].count().sort_values(ascending=False))


# In[ ]:


# Compare normal vs arrhythmia patients on key measurements
print(df.groupby("status")[["age", "heart_rate", "qrs_duration", "qt_interval"]].mean().round(1))


# In[ ]:


#groupby two columns, with named aggregations
grouped = df.groupby(["status", "sex"]).agg(
    patients=("patient_id", "count"),
    mean_age=("age", "mean"),
    mean_hr=("heart_rate", "mean"),
    max_qrs=("qrs_duration", "max"),).round(1)
print(grouped)



# In[ ]:


#groupby on the melted (long) data : mean of each interval by status
print(long_df.groupby(["measurement", "status"])["ms"].mean().unstack().round(1))


# In[ ]:




