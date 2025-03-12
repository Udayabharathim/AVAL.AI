# import os
# # import os

# # image_dir = "C:/Users/Elitebook 840 G6/Documents/AVAL2/data/pigmentation/train"
# # label_dir = image_dir  # Assuming labels are stored in the same folder

# # image_files = {f.split('.')[0] for f in os.listdir(image_dir) if f.endswith('.jpg')}
# # label_files = {f.split('.')[0] for f in os.listdir(label_dir) if f.endswith('.txt')}

# # missing_labels = image_files - label_files
# # missing_images = label_files - image_files

# # print("Images without labels:", missing_labels)
# # print("Labels without images:", missing_images)

# import os
# import glob

# # Define your dataset folder
# labels_path = "C:/Users/Elitebook 840 G6/Documents/AVAL2/data/pigmentation/train/labels"

# # Get all .txt label files
# label_files = glob.glob(os.path.join(labels_path, "*.txt"))

# # Iterate through each label file
# for file in label_files:
#     with open(file, "r") as f:
#         lines = f.readlines()

#     fixed_lines = []
#     for line in lines:
#         parts = line.strip().split()
#         if len(parts) == 5:  # Ensure correct format
#             class_id = parts[0]
#             x_center, y_center, width, height = map(float, parts[1:])

#             # Fix negative values and values >1
#             x_center = max(0, min(1, x_center))
#             y_center = max(0, min(1, y_center))
#             width = max(0, min(1, width))
#             height = max(0, min(1, height))

#             fixed_lines.append(f"{class_id} {x_center} {y_center} {width} {height}\n")

#     # Overwrite file with fixed values
#     with open(file, "w") as f:
#         f.writelines(fixed_lines)

# print("✅ Labels fixed successfully!")
import os

labels_dir = r"C:\Users\Elitebook 840 G6\Documents\AVAL2\data\hirsuitsm\train\labels"

for file in os.listdir(labels_dir):
    if file.endswith(".txt"):
        file_path = os.path.join(labels_dir, file)
        with open(file_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 6:  # If it has 6 columns, remove the last one
                new_lines.append(" ".join(parts[:5]) + "\n")
            else:
                new_lines.append(line)

        with open(file_path, "w") as f:
            f.writelines(new_lines)

print("Label files fixed!")
