import os

def update_labels(label_dir, class_mapping):
    """
    Updates YOLO-format label files to reflect a new class mapping, where labels
    are stored as indices, and merges classes.

    Args:
        label_dir (str): Directory containing the YOLO-format .txt label files.
        class_mapping (dict): A dictionary mapping old class indices to new class indices.
                              If an old index should be merged with another, map it to the target index.
                              Example:
                              {
                                  0: 0,  # Old index 0 maps to new index 0
                                  1: 1,  # Old index 1 maps to new index 1
                                  2: 0,  # Old index 2 maps to new index 0 (MERGED)
                                  9: 10 # Old index 9 maps to new index 10
                              }
    """

    for filename in os.listdir(label_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(label_dir, filename)
            updated_lines = []

            with open(filepath, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 0:  # Handle empty lines
                        try:
                            old_class_index = int(parts[0])
                            if old_class_index in class_mapping:
                                new_class_index = class_mapping[old_class_index]
                                updated_line = f"{new_class_index} {' '.join(parts[1:])}"
                                updated_lines.append(updated_line)
                            else:
                                print(f"Warning: Old class index {old_class_index} not found in mapping in file {filename}. Skipping line.")
                        except ValueError:
                            print(f"Warning: Invalid class index in line: {line.strip()} in file {filename}. Skipping line.")

            # Write the updated labels back to the file
            with open(filepath, "w") as f:
                f.write("\n".join(updated_lines))


# Example usage (replace with your actual paths and mapping)

label_directory = r"C:\Users\suren\Desktop\output\output\val\labels"  # The directory containing your YOLO .txt label files

# Class Indices based on the order you mentioned in your prompt
# Before classes: 'Arc Suit 12 Kcal/m2', 'Arc Suit 45 Kcal/m2', 'Arc Suit 8.8 Kcal/m2', 'Face Shield', 'Hand Gloves', 'No Arc Suit', 'No Face Shield', 'No Hand Gloves', 'No Safety Helmet', 'No Safety Shoes', 'Safety Helmet', 'Safety Shoes'
# After Classes: 'Arc Suit 12 Kcal/m2', 'Arc Suit 45 Kcal/m2', 'Arc Suit 8.8 Kcal/m2', 'Face Shield', 'Hand Gloves', 'No Arc Suit', 'No Face Shield', 'No Hand Gloves', 'No Reflective Jacket', 'No Safety Helmet', 'No Safety Shoes', 'Reflective Jacket', 'Safety Helmet', 'Safety Shoes'
# Change:  Merge classes: 'Arc Suit 12 Kcal/m2' and 'Arc Suit 8.8 Kcal/m2' --> index 0

class_mapping = {
    0: 0,  # 'Arc Suit 12 Kcal/m2' remains 0
    1: 1,  # 'Arc Suit 45 Kcal/m2' remains 1
    2: 0,  # 'Arc Suit 8.8 Kcal/m2' becomes 0 (MERGED with 'Arc Suit 12 Kcal/m2')
    3: 3,  # 'Face Shield' remains 3
    4: 4,  # 'Hand Gloves' remains 4
    5: 5,  # 'No Arc Suit' remains 5
    6: 6,  # 'No Face Shield' remains 6
    7: 7,  # 'No Hand Gloves' remains 7
    8: 9,  # 'No Safety Helmet' becomes 9
    9: 10, # 'No Safety Shoes' becomes 10
    10: 12, # 'Safety Helmet' becomes 12
    11: 13 # 'Safety Shoes' becomes 13
}

update_labels(label_directory, class_mapping)
print("Labels updated successfully!")