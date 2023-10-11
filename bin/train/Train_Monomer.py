import sys
import argparse
import os
from os import listdir
from os.path import isfile, isdir, join
import numpy as np
import pandas as pd
import multiprocessing
import sys
sys.path.insert(0, "./")

from QA_File.dataset import set_QA_Dloader
import QA_File
import torch.optim as optim
import os
from torch.utils.data import Dataset, DataLoader
import torch

from QA_File.QA_utils.folding import process_prediction
import warnings
warnings.filterwarnings("ignore")
def main():

    parser = argparse.ArgumentParser(description="predictor network train")
    
    parser.add_argument("input",                          action="store",                              help="input dir path.")
    
    parser.add_argument("folder",                         action="store",                              help="Location of folder to save checkpoints to.")
    
    parser.add_argument("--epoch",       "-e",            action="store",  type=int,   default=150)

    parser.add_argument("--multi_dir",   "-multi_dir",    action="store_true",         default=False,  help="Run with multiple direcotory sources")
    
    parser.add_argument("--num_blocks",  "-numb",         action="store",  type=int,   default=3,      help="# reidual blocks")
    
    parser.add_argument("--num_filters", "-numf",         action="store",  type=int,   default=128,    help="# of base filter size in residual blocks")
    
    parser.add_argument("--size_limit",  "-size_limit",   action="store",  type=int,   default=300,    help="protein size limit")
    
    parser.add_argument("--decay","-d",                   action="store",  type=float, default=0.99,   help="Decay rate for learning rate (Default: 0.99)")
    
    parser.add_argument("--base","-b",                    action="store",  type=float, default=0.001, help="Base learning rate (Default: 0.0005)")
    
    parser.add_argument("--debug",       "-debug",        action="store_true",         default=False,  help="Debug mode (Default: False)")
    
    parser.add_argument("--silent",      "-s",            action="store_true",         default=False,  help="Run in silent mode (Default: False)")
   
    args = parser.parse_args()

    script_dir = os.path.dirname(__file__)                  
    print("script_dir:",script_dir)
    base = join(script_dir, "data_list")                       

    epochs = args.epoch
    base_learning_rate = args.base                          
    decay = args.decay                                      
    loss_weight = [0.1, 0.1]                                
    validation = True                                      
    name = args.folder
    lengthmax = args.size_limit
    
    fea_dir=join(args.input, "features")
    pdb_dir=join(args.input, "pdbs")
    seq_dir=join(args.input, "seq_emb")
    str_dir=join(args.input, "seq_s_emb")

    if not args.silent: print("Loading samples")           
    proteins = np.load(join(base, "train_decoys.npy"))
    proteins = ["1a7tA", "1a8rA", "1pyoB", "1pzlA"]
    
    if args.debug: proteins = proteins[:50]               
    train_decoys = QA_File.DecoyDataset(targets = proteins,
                                           lengthmax = lengthmax,
                                           multi_dir = args.multi_dir,
                                           root_dir = fea_dir,
                                           pdb_dirs = pdb_dir,
                                           emb_path = seq_dir,
                                           structure_emb = str_dir
                                           )

    train_dataloader = set_QA_Dloader(train_decoys, num_works=8,batch=1)


    proteins = np.load(join(base, "valid_decoys.npy"))
    proteins = ["1a7tA", "1a8rA", "1pyoB", "1pzlA"]
    if args.debug: proteins = proteins[:50]

    valid_decoys = QA_File.DecoyDataset(targets = proteins,
                                           lengthmax = lengthmax,
                                           multi_dir = args.multi_dir,
                                           root_dir = fea_dir,
                                           pdb_dirs = pdb_dir,
                                           emb_path = seq_dir,
                                           structure_emb = str_dir
                                           ) 
    valid_dataloader = set_QA_Dloader(valid_decoys,num_works=8,batch=1)

    # Load the model if needed
    if not args.silent: print("instantitate a model")       
    net = QA_File.QA(num_channel  = args.num_filters)

    get_parameter_number(net)
    rdevreModel = False

    if isdir(args.folder):                                  
        if not args.silent: print("checkpoint")
        if not os.path.exists(join(name, "model.pkl")):
            checkpoint = torch.load(join(name, "model.pkl"), map_location='cpu')
            net.load_state_dict(checkpoint["model_state_dict"]) 

            epoch = checkpoint["epoch"]+1
            train_loss = checkpoint["train_loss"]
            valid_loss = checkpoint["valid_loss"]
            best_models = checkpoint["best_models"]
            if not args.silent: print("Restarting at epoch", epoch)
            assert(len(train_loss["loss"]) == epoch)
            assert(len(valid_loss["loss"]) == epoch)
            rdevreModel = True
            epoch = 0
            train_loss = {"loss": [], "coords_loss": [], "bondlen_loss": [], "p_lddt_loss": [], "dev": [], "mask": []}
            valid_loss = {"loss": [], "coords_loss": [], "bondlen_loss": [], "p_lddt_loss": [], "dev": [], "mask": []}
            best_models = []
        else:
            epoch = 0
            train_loss = {"loss":[], "coords_loss":[], "bondlen_loss":[], "p_lddt_loss":[],"dev":[],"mask":[]}
            valid_loss = {"loss":[], "coords_loss":[], "bondlen_loss":[], "p_lddt_loss":[],"dev":[],"mask":[]}
            best_models = []
    else:
        if not args.silent: print("Training")              
        epoch = 0
        train_loss = {"loss":[], "coords_loss":[], "bondlen_loss":[], "p_lddt_loss":[],"dev":[],"mask":[]}
        valid_loss = {"loss":[], "coords_loss":[], "bondlen_loss":[], "p_lddt_loss":[],"dev":[],"mask":[]}
        best_models = []

        if not isdir(name):
            if not args.silent: print("Creating new dir", name)
            os.mkdir(name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
    net.to(device)                                        

    optimizer = optim.AdamW(net.parameters(), lr=0.001)
    if rdevreModel:                                        
        checkpoint = torch.load(join(name, "model.pkl"))
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Loop over the dataset multiple times
    start_epoch = epoch                                     
    for epoch in range(start_epoch, epochs):                

        # Update the learning rate
        lr = base_learning_rate*np.power(decay, epoch)     
        print("lr = ", lr)                                 

      

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        
       
        net.train(True)
        temp_loss = {"loss":[], "coords_loss":[], "bondlen_loss":[], "p_lddt_loss":[],"dev":[],"mask":[]}
        for i, data in enumerate(train_dataloader):

            idx, val, f1d, f2d, dev, dev_1hot, mask = data["idx"], data["val"], data["1d"], data["2d"],\
                                                        data["deviation"], data["deviation_1hot"], data["mask"]
            idx = idx[0].long().to(device)
            val = val[0].to(device)
            f1d = f1d[0].to(device)
            f2d = f2d[0].to(device)
            dev_true = dev[0].to(device)  
            dev_1hot_true = dev_1hot[0].to(device)
            mask_true = mask[0].to(device)  

           
            node_emb = data["node_emb"].to(device)
            model_coords = data["model_coords"].to(device)
            native_coord = data["native_coord"][0].to(device)

           
            # Zero the parameter gradients
            lddt_true = QA_File.calculate_LDDT(dev_1hot_true[0], mask_true[0]).to(device)
            output, mask_logits, deviation_logits = net(idx, val, f1d, f2d, node_emb, model_coords,native_coord,lddt_true)

            coords_loss = output.coords_loss
            bondlen_loss = output.bondlen_loss
            p_lddt_loss = output.p_lddt_loss
            Esto_Loss = torch.nn.CrossEntropyLoss()   
            Mask_Loss = torch.nn.BCEWithLogitsLoss()    

            dev_loss = Esto_Loss(deviation_logits, dev_true)    
            mask_loss = Mask_Loss(mask_logits[0], mask_true[0])
    
            loss = coords_loss+bondlen_loss+p_lddt_loss + loss_weight[0]*dev_loss + loss_weight[1]*mask_loss
            

            loss.backward()                            
            optimizer.step()                           
            optimizer.zero_grad()  
            # Get training loss
            temp_loss["loss"].append(loss.cpu().detach().numpy())
            temp_loss["coords_loss"].append(coords_loss.cpu().detach().numpy())
            temp_loss["bondlen_loss"].append(bondlen_loss.cpu().detach().numpy())
            temp_loss["p_lddt_loss"].append(p_lddt_loss.cpu().detach().numpy())
            temp_loss["dev"].append(dev_loss.cpu().detach().numpy())
            temp_loss["mask"].append(mask_loss.cpu().detach().numpy())

            # Display training results
            print("training results:")
            sys.stdout.write(
                        "\rEpoch: [%2d/%2d], Batch: [%2d/%2d], "
                        "loss: %.4f-%.4f, coords_loss: %.4f-%.4f, "
                        "bondlen_loss: %.4f-%.4f, p_lddt_loss: %.4f-%.4f, "
                        "dev-loss: %.4f-%.4f, mask: %.4f-%.4f"
                        % (epoch + 1, epochs, i + 1, len(valid_decoys),
                           temp_loss["loss"][-1],np.mean(temp_loss["loss"]),
                           temp_loss["coords_loss"][-1], np.mean(temp_loss["coords_loss"]),
                           temp_loss["bondlen_loss"][-1],np.mean(temp_loss["bondlen_loss"]),
                           temp_loss["p_lddt_loss"][-1],np.mean(temp_loss["p_lddt_loss"]),
                           temp_loss["dev"][-1], np.mean(temp_loss["dev"]),
                           temp_loss["mask"][-1],np.mean(temp_loss["mask"])))

        # Save the loss
        train_loss["loss"].append(np.array(temp_loss["loss"]))
        train_loss["coords_loss"].append(np.array(temp_loss["coords_loss"]))
        train_loss["bondlen_loss"].append(np.array(temp_loss["bondlen_loss"]))
        train_loss["p_lddt_loss"].append(np.array(temp_loss["p_lddt_loss"]))
        train_loss["dev"].append(np.array(temp_loss["dev"]))
        train_loss["mask"].append(np.array(temp_loss["mask"]))

        if validation:                                  
            net.eval()                                 
            temp_loss = {"loss":[], "coords_loss":[], "bondlen_loss":[], "p_lddt_loss":[],"dev":[],"mask":[]}
           
            with torch.no_grad():                       #
                for i, data in enumerate(valid_dataloader):
                    # Get the data, Hardcoded transformation for whatever reasons.
                    idx, val, f1d, f2d, dev, dev_1hot, mask = data["idx"], data["val"], data["1d"], data["2d"], \
                                                              data["deviation"], data["deviation_1hot"], data["mask"]
                    idx = idx[0].long().to(device)
                    val = val[0].to(device)
                    f1d = f1d[0].to(device)
                    f2d = f2d[0].to(device)
                    dev_true = dev[0].to(device)  
                    dev_1hot_true = dev_1hot[0].to(device)
                    mask_true = mask[0].to(device)  
                    
                    geo_dist = None
                    geo_ori = None
                    # coord
                    node_emb = data["node_emb"].to(device)
                    model_coords = data["model_coords"].to(device)
                    native_coord = data["native_coord"][0].to(device)
                   
                    lddt_true = QA_File.calculate_LDDT(dev_1hot_true[0], mask_true[0]).to(device)
                    output, mask_logits, deviation_logits = net(idx, val, f1d, f2d, node_emb, model_coords, native_coord, lddt_true)

                    coords_loss = output.coords_loss
                    bondlen_loss = output.bondlen_loss
                    p_lddt_loss = output.p_lddt_loss
                    # loss = output.loss

                    Esto_Loss = torch.nn.CrossEntropyLoss()  
                    Mask_Loss = torch.nn.BCEWithLogitsLoss()  
                    dev_loss = Esto_Loss(deviation_logits, dev_true)  
                    mask_loss = Mask_Loss(mask_logits[0], mask_true[0])
                    loss = coords_loss + bondlen_loss + p_lddt_loss + loss_weight[0] * dev_loss + loss_weight[1] * mask_loss

                    # Get validation loss
                    temp_loss["loss"].append(loss.cpu().detach().numpy())
                    temp_loss["coords_loss"].append(coords_loss.cpu().detach().numpy())
                    temp_loss["bondlen_loss"].append(bondlen_loss.cpu().detach().numpy())
                    temp_loss["p_lddt_loss"].append(p_lddt_loss.cpu().detach().numpy())
                    temp_loss["dev"].append(dev_loss.cpu().detach().numpy())
                    temp_loss["mask"].append(mask_loss.cpu().detach().numpy())

                    # Display validation results
                    print("validation results:")
                    sys.stdout.write(
                        "\rEpoch: [%2d/%2d], Batch: [%2d/%2d], "
                        "loss: %.4f-%.4f, coords_loss: %.4f-%.4f, "
                        "bondlen_loss: %.4f-%.4f, p_lddt_loss: %.4f-%.4f, "
                        "dev-loss: %.4f-%.4f, mask: %.4f-%.4f"
                        % (epoch + 1, epochs, i + 1, len(valid_decoys),
                           temp_loss["loss"][-1], np.mean(temp_loss["loss"]),
                           temp_loss["coords_loss"][-1], np.mean(temp_loss["coords_loss"]),
                           temp_loss["bondlen_loss"][-1], np.mean(temp_loss["bondlen_loss"]),
                           temp_loss["p_lddt_loss"][-1], np.mean(temp_loss["p_lddt_loss"]),
                           temp_loss["dev"][-1], np.mean(temp_loss["dev"]),
                           temp_loss["mask"][-1], np.mean(temp_loss["mask"])))

            valid_loss["loss"].append(np.array(temp_loss["loss"]))
            valid_loss["coords_loss"].append(np.array(temp_loss["coords_loss"]))
            valid_loss["bondlen_loss"].append(np.array(temp_loss["bondlen_loss"]))
            valid_loss["p_lddt_loss"].append(np.array(temp_loss["p_lddt_loss"]))
            valid_loss["dev"].append(np.array(temp_loss["dev"]))
            valid_loss["mask"].append(np.array(temp_loss["mask"]))

            # Saving the model if needed.
            if name != "" and validation:               

                folder = name
                # Name of ranked models. I know it is not optimal way to do it but the easiest fix is this.
                name_map = ["best.pkl", "second.pkl", "third.pkl", "fourth.pkl", "fifth.pkl"]

                new_model = (epoch, np.mean(valid_loss["p_lddt_loss"][-1]))   
                new_best_models = best_models[:]                        
                new_best_models.append(new_model)
                new_best_models.sort(key=lambda x: x[1])

                temp = new_best_models[:len(name_map)]
                new_best_models = [(temp[i][0], temp[i][1], name_map[i]) for i in range(len(temp))]

                # Saving and moving
                for i in range(len(new_best_models)):
                    m, performance, filename = new_best_models[i]
                    if m in [j[0] for j in best_models]:
                        index = [j[0] for j in best_models].index(m)
                        command = "mv %s %s"%(join(folder, best_models[index][2]), join(folder, "temp_"+new_best_models[i][2]))
                        os.system(command)
                    else:
                         torch.save({
                            'epoch': epoch,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'train_loss': train_loss,
                            'valid_loss': valid_loss,
                        }, join(folder, "temp_"+new_best_models[i][2]))

                # Renaming
                for i in range(len(new_best_models)):
                    command = "mv %s %s"%(join(folder, "temp_"+name_map[i]), join(folder, name_map[i]))
                    os.system(command)                                  

                # Update best list
                best_models = new_best_models

            # Save all models
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'valid_loss': valid_loss,
                    'best_models' : best_models
                    }, join(name, "model.pkl"))



def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print('+++Trainable+++',trainable_num)
    return {'+++Total+++': total_num, '+++Trainable+++': trainable_num}

if __name__== "__main__":
    main()
