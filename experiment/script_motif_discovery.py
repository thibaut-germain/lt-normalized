import subprocess

import sys
sys.path.append("./src")
import os 
import json 

EXP_ID = 0

distances = ["LTNormalizedEuclidean", "NormalizedEuclidean"]
algo= "MatrixProfile"
folders = os.listdir("./dataset/")
folders.remove(".DS_Store")
folders.remove("computation-time")

#for f in folders: 
#    config_path = "./experiment/"+ f + "/configs/base_configs.json"
#    with open(config_path,"r") as fl: 
#        configs = json.load(fl)
#    config = json.loads(configs[algo])[0]
#    lst = []
#    for dist in distances: 
#        t_config = config.copy()
#        t_config["distance_name"] = dist
#        t_config["radius_ratio"] = 3
#        lst.append(t_config)
#    configs = {algo: json.dumps(lst)}
#
#    save_path = "./experiment/lt-normalized/configs/" + f"{f}_configs.json"
#    with open(save_path, "w") as fl: 
#        json.dump(configs,fl)
    

if __name__ == "__main__": 

    command = ""
    for f in folders: 
        data_path = "./dataset/" + f +"/dataset.pkl"
        label_path = "./dataset/" + f +"/labels.pkl"
        config_path = "./experiment/configs/" + f"{f}_configs.json"
        result_path = f"./experiment/results/{f}_"
        indv_command = f"python ./experiment/algo_configs_exp.py {algo} {config_path} {data_path} {label_path} {result_path} {EXP_ID} & "
        command += indv_command

    subprocess.run(command, shell=True)

