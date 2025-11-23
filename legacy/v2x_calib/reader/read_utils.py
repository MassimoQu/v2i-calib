import json
import yaml

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)
    
def read_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)
