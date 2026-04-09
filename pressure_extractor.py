import json
import argparse

def extract_pressure_data(json_file):
    i = 0
    j = 10
    k = 0
    initial_timestamp = 0
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    pressure_data = []
    mod_pressure_data = []
    for entry in data:
        if 'value' in entry and i > 10:
            pressure_data.append((entry['value'], entry['timestamp']))
        i = i + 1
    
    for val in pressure_data:
        j = j + 1
        
        if val[0] != 0 and k == 0:
            initial_timestamp = val[1]
            mod_pressure_data.append((val[0], val[1] - initial_timestamp))
            k = 1
        elif k != 0:
            mod_pressure_data.append((val[0], val[1] - initial_timestamp))

    with open(json_file.replace('.json', '_corrected.json'), 'w') as file:
        json.dump(mod_pressure_data, file, indent=4)
    return mod_pressure_data



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="blah")
    parser.add_argument("filename", help="input JSON file name")

    args = parser.parse_args()
    extract_pressure_data(args.filename)
