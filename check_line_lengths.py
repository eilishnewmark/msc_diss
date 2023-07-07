def check_line_lengths(char_file, POS_file):
    with open(char_file, "r") as char_file:
        with open(POS_file, "r") as POS_file:
            char_lines = char_file.readlines()
            POS_lines = POS_file.readlines()
    assert len(char_lines) == len(POS_lines), "files are of different lengths!"

    line = 0
    for chars, POS in zip(char_lines, POS_lines):
        line += 1
        chars = chars.rstrip().split()
        POS = POS.strip(" +\n").split(" + ")
        assert len(chars) == len(POS), f"different no. of words in line! line {line}"
        for char_word, POS_word in zip(chars, POS):
            if len(char_word) != len(POS_word.split()):
                print("line: ", line, "chars: ", char_word, "POS: ", POS_word)

check_line_lengths("WHD_data/data/WHD_src.txt", "WHD_data/data/WHD_POS.txt")