import os
import subprocess

pages_dir_path = "/Users/rmencis/RUG/Handwriting_Recognition/finaltest"
words_dir_path = "/Users/rmencis/RUG/Handwriting_Recognition/finaltest"
output_dir_path = "/Users/rmencis/RUG/Handwriting_Recognition/finaltest/output/ctc"

page_file_names = [f for f in os.listdir(pages_dir_path) if f.endswith(".ppm")]

for page_file_name in page_file_names:
    print("==========================================================================================================")
    print(page_file_name)

    words_file_name = page_file_name.replace(".ppm",".words")

    page_file_path = os.path.join(pages_dir_path,page_file_name)
    words_file_path = os.path.join(words_dir_path,words_file_name)
    output_file_path = os.path.join(output_dir_path,words_file_name)

    subprocess.run(["python3", "recognizer.py","-c",page_file_path, words_file_path, output_file_path])

