import yaml

def read_yml(PATH):
    # Charger le fichier YAML
    with open(PATH, 'r') as file:
        data = yaml.safe_load(file)
    
    for key in data:
        try:
            data[key] = eval(data[key])
        except:
            pass

    return data