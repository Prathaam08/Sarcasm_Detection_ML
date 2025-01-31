# Data/Sarcasm_Headlines_Dataset.json
import re

with open('Data/Sarcasm_Headlines_Dataset_v2.json', 'r') as file:
    content = file.read()

# Use regular expression to insert commas between adjacent JSON objects
fixed_content = re.sub(r'}\s*{', '},{', content)

# Optionally, wrap the content in an array if necessary
fixed_content = f'[{fixed_content}]'

with open('fixed_file.json', 'w') as file:
    file.write(fixed_content)

print("File has been fixed and saved as 'fixed_file.json'.")
