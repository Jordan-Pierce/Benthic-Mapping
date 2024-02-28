import os
import tator

# -----------------------------------------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------------------------------------

root = "./Data"
os.makedirs(root, exist_ok=True)

test_output = f"{root}/output.txt"

with open(test_output, 'w') as file:
    # Write content to the file
    file.write('This was output from the container!')

print(f"TEST: {test_output} {os.path.exists(test_output)}")