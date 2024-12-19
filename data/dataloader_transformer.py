from data.dataloader_Fishbowl import FishBowl
from data.dataloader_UOAIS import Fusion_UOAIS
from data.dataloader_OSD import Fusion_OSD
from data.dataloader_UOAIS_allvm import Fusion_UOAIS_ALLVM
from data.dataloader_OSD_allvm import Fusion_OSD_ALLVM

def load_dataset(config, args, mode):
    if mode=="train":
        if args.dataset=="UOAIS":
            train_dataset = Fusion_UOAIS(config, mode='train')
            test_dataset = Fusion_UOAIS(config, mode='test')
        elif args.dataset=="OSD":
            train_dataset = Fusion_OSD(config, mode='train')
            test_dataset = Fusion_OSD(config, mode='test')
        return train_dataset, test_dataset 
    else:
        if args.dataset=="UOAIS":
            test_dataset = Fusion_UOAIS(config, mode='test')
        elif args.dataset=="OSD":
            test_dataset = Fusion_OSD(config, mode='test')
        elif args.dataset=="UOAIS_ALLVM":
            test_dataset = Fusion_UOAIS_ALLVM(config, mode='test')
        elif args.dataset=="OSD_ALLVM":
            test_dataset = Fusion_OSD_ALLVM(config, mode='test')
        return test_dataset