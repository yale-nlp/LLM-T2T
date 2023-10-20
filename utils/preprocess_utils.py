import json

def read_json(data_path):
    with open(data_path, 'r', encoding='utf-8') as load_file:
        data_json = json.load(load_file)

    return data_json