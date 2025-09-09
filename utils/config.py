import yaml

properties = 'api.yaml'

def getAPIValue(key):
    with open('../api.yaml') as f:
        prop = yaml.load(f, Loader=yaml.FullLoader)
        return prop.get(key)
