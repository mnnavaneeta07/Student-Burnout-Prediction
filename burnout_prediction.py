# ============================================================
# STUDENT BURNOUT PREDICTION SYSTEM
# Version 2 - Part 1
# ============================================================

# -------------------- IMPORT LIBRARIES --------------------

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# -------------------- LOAD DATASET --------------------

print("=" * 60)
print("      STUDENT BURNOUT PREDICTION SYSTEM")
print("=" * 60)

print("\nLoading Dataset...")

df = pd.read_csv("student_mental_health_burnout.csv")

print("✓ Dataset Loaded Successfully!")

# -------------------- DATASET DETAILS --------------------

print("\nDataset Shape :", df.shape)

print("\nFirst 5 Records")
print(df.head())

print("\nColumn Names")
print(df.columns.tolist())

print("\nChecking Missing Values...")

missing = df.isnull().sum()

print(missing)

if missing.sum() == 0:
    print("\n✓ No Missing Values Found")
else:
    print("\nMissing Values Exist!")

# -------------------- LABEL ENCODING --------------------

print("\nEncoding Dataset...")

encoders = {}

categorical_columns = [
    "gender",
    "course",
    "year",
    "stress_level",
    "sleep_quality",
    "internet_quality",
    "burnout_level"
]

for column in categorical_columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    encoders[column] = encoder

print("✓ Encoding Completed")

# -------------------- FEATURES & TARGET --------------------

X = df.drop("burnout_level", axis=1)
y = df["burnout_level"]

print("\nFeatures :", X.shape)
print("Target   :", y.shape)

# -------------------- TRAIN TEST SPLIT --------------------

print("\nSplitting Dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("✓ Train-Test Split Completed")

# -------------------- TRAIN MODEL --------------------

print("\nTraining Decision Tree Model...")

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

print("✓ Model Trained Successfully")

# -------------------- MODEL ACCURACY --------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy : {accuracy*100:.2f}%")

# -------------------- AVAILABLE OPTIONS --------------------

print("\nAvailable Options")

print("-" * 60)

for column in [
    "gender",
    "course",
    "year",
    "stress_level",
    "sleep_quality",
    "internet_quality"
]:
    print(f"{column.upper():20} : {list(encoders[column].classes_)}")

print("-" * 60)

print("\nModel is Ready for Prediction!")
# ============================================================
# VERSION 2 - PART 2
# USER INPUT & BURNOUT PREDICTION
# ============================================================

# ---------- FUNCTION TO VALIDATE CATEGORICAL INPUT ----------

def get_valid_input(column_name):

    options = list(encoders[column_name].classes_)

    while True:
        print(f"\nAvailable {column_name.replace('_',' ').title()} :")
        print(", ".join(options))

        value = input(f"Enter {column_name.replace('_',' ').title()} : ").strip()

        if value in options:
            return value

        print("Invalid input! Please choose one of the above options.")


# ============================================================
# USER INPUT
# ============================================================

print("\n" + "="*60)
print("ENTER STUDENT DETAILS")
print("="*60)

student_id = int(input("Student ID : "))
age = int(input("Age : "))

gender = get_valid_input("gender")
course = get_valid_input("course")
year = get_valid_input("year")

daily_study_hours = float(input("Daily Study Hours : "))
daily_sleep_hours = float(input("Daily Sleep Hours : "))
screen_time_hours = float(input("Screen Time Hours : "))

stress_level = get_valid_input("stress_level")

anxiety_score = int(input("Anxiety Score (1-10): "))
depression_score = int(input("Depression Score (1-10): "))
academic_pressure_score = int(input("Academic Pressure Score (1-10): "))
financial_stress_score = int(input("Financial Stress Score (1-10): "))
social_support_score = int(input("Social Support Score (1-10): "))

physical_activity_hours = float(input("Physical Activity Hours : "))

sleep_quality = get_valid_input("sleep_quality")

attendance_percentage = float(input("Attendance Percentage : "))
cgpa = float(input("CGPA : "))

internet_quality = get_valid_input("internet_quality")


# ============================================================
# ENCODE USER INPUT
# ============================================================

gender = encoders["gender"].transform([gender])[0]
course = encoders["course"].transform([course])[0]
year = encoders["year"].transform([year])[0]
stress_level = encoders["stress_level"].transform([stress_level])[0]
sleep_quality = encoders["sleep_quality"].transform([sleep_quality])[0]
internet_quality = encoders["internet_quality"].transform([internet_quality])[0]


# ============================================================
# CREATE DATAFRAME
# ============================================================

user_data = pd.DataFrame([[

    student_id,
    age,
    gender,
    course,
    year,
    daily_study_hours,
    daily_sleep_hours,
    screen_time_hours,
    stress_level,
    anxiety_score,
    depression_score,
    academic_pressure_score,
    financial_stress_score,
    social_support_score,
    physical_activity_hours,
    sleep_quality,
    attendance_percentage,
    cgpa,
    internet_quality

]], columns=X.columns)


# ============================================================
# PREDICTION
# ============================================================

prediction = model.predict(user_data)

result = encoders["burnout_level"].inverse_transform(prediction)

print("\n" + "="*60)
print("PREDICTION RESULT")
print("="*60)

print(f"\nPredicted Burnout Level : {result[0]}")

if result[0] == "Low":
    print("\nRisk Level : LOW")
    print("The student appears to have a low risk of burnout.")
    print("Continue maintaining healthy study and lifestyle habits.")

elif result[0] == "Medium":
    print("\nRisk Level : MEDIUM")
    print("The student has a moderate risk of burnout.")
    print("Better time management and regular breaks are recommended.")

elif result[0] == "High":
    print("\nRisk Level : HIGH")
    print("The student has a high risk of burnout.")
    print("Stress management and counselling are recommended.")

print("\n" + "="*60)
print("THANK YOU")
print("Student Burnout Prediction System")
print("="*60)