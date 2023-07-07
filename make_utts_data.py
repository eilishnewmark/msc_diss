import re

with open('WHD_data/data/WHD_src.txt', "r") as in_file, open("WHD_data/utts.data", "w") as out_file:
    sentences = in_file.readlines()
    i = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence == "":
            continue
        else:
            i += 1
            out_file.write(f'( eilish_{i} "{sentence}" )\n')