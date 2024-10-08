import pandas as pd
from collections import defaultdict

# Load the Excel file
file_path = "./JD/JD_Data_Analyst.xlsx"
df = pd.read_excel(file_path)

# Extract the "Skills" column
skills_column = df["Skills"]

# Create a list to store unique skills
skills_list = []

# Process each row in the Skills column
for skills in skills_column:
    if isinstance(skills, str):  # Check if the entry is a string
        # Split skills by comma and strip any extra spaces
        skill_list = [skill.strip() for skill in skills.split(",")]
        # Add each skill to the list if it's not already present
        for skill in skill_list:
            if skill and skill not in skills_list:  # Avoid duplicates
                skills_list.append(skill)

# Print the list of unique skills
print(skills_list)