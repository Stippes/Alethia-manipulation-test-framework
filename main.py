from scripts.input_parser import parse_json_chat, parse_txt_chat, standardize_format

raw1 = parse_json_chat("data/example.json")
std1 = standardize_format(raw1)
print("Standardized format json", std1)

raw2 = parse_txt_chat("data/example.txt")
std2 = standardize_format(raw2)
print("Standardized format txt",std2)
