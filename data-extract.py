import os 
import lzma
from tqdm import tqdm

def xz_files_in_dir(directory):
    files = []
    # Walk through directory and all subdirectories
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            # Check if file is an .xz file
            if filename.endswith(".xz"):
                # Get the full path to the file
                full_path = os.path.join(root, filename)
                if os.path.isfile(full_path):
                    # You can either store the full path or just the filename
                    files.append(full_path)  # or files.append(filename) if you only want the filename
    return files


folder_path= "/Users/apple/Documents/openwebtext/subsets"
output_file_train = "output_train.txt"
output_file_val = "output_val.txt"
vocab_file = "vocab.txt"

files = xz_files_in_dir(folder_path)
total_files = len(files)


split_index = int(total_files * 0.9)
files_train = files [:split_index]
files_val = files [split_index:]

vocab = set()

                    


with open(output_file_train, "w", encoding  = "utf-8") as outfile: 
     for filename in tqdm(files_train, total = len(files_train)):
          file_path = os.path.join(folder_path,filename)
          with lzma.open(file_path, "rt", encoding = "utf-8") as infile:
               text = infile.read()
               outfile.write(text)
               characters = set(text)
               vocab.update(characters)


 
with open(output_file_val, "w", encoding = "utf-8") as outfile:
     for filename in tqdm(files_val, total= len(files_val)):
          file_path = os.path.join(folder_path, filename)
          with lzma.open(file_path, "rt" , encoding = "utf-8") as infile:
               text = infile.read()
               outfile.write(text)
               characters = set(text)
               vocab.update(characters)

 
with open('vocab_file', 'w', encoding = "utf-8") as vfile:
     for char in sorted(vocab):
          vfile.write(char + "\n")