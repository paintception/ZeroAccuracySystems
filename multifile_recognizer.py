import os
import subprocess

pages_dir_path = "/Users/rmencis/RUG/Handwriting_Recognition/pages/Stanford"
words_dir_path = "/Users/rmencis/RUG/Handwriting_Recognition/code/ZeroAccuracySystems/labels"
output_dir_path = "/Users/rmencis/RUG/Handwriting_Recognition/output_words"

page_file_names = [f for f in os.listdir(pages_dir_path) if f.endswith(".jpg")]

for page_file_name in page_file_names:
    print("==========================================================================================================")
    print(page_file_name)

    words_file_name = page_file_name.replace(".jpg",".words")

    page_file_path = os.path.join(pages_dir_path,page_file_name)
    words_file_path = os.path.join(words_dir_path,words_file_name)
    output_file_path = os.path.join(output_dir_path,words_file_name)

    if (os.path.exists(words_file_path)):
        subprocess.run(["python3", "recognizer.py","-c",page_file_path, words_file_path, output_file_path])

