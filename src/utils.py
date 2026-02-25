import yaml

class Utils:
    def load_config(self)-> dict:
        #Loads the config.yaml file and returns the config as a dictionary
        config_path = "./config.yaml"
        config = None
        try:
            with open(config_path,'r') as file:
                config = yaml.safe_load(file)
        except Exception as e:
            print(f"An error occured while loading the file {e}")
        return config