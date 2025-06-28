import torch
from torch.utils import data
from thop import profile
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

import model
import utils
import utils.func
import utils.mm_loader
import utils.metric

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    
    #--------------------------------------------------------------------------------------------
    utils.func.set_seed(cfg.model.training_settings.seed)

    device = torch.device(cfg.model.training_settings.device if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available())

    #--------------------------------------------------------------------------------------------
    train_sets = utils.mm_loader.mm_dataset(cfg,phase ='train')
    test_sets = utils.mm_loader.mm_dataset(cfg,phase ='test')

    print("train : {} , test : {}".format(len(train_sets),len(test_sets)))
    train_loader = data.DataLoader(
        train_sets,
        cfg.model.training_settings.train_batch_size, 
        shuffle=False, 
        drop_last=True,
        num_workers=cfg.model.training_settings.num_workers
    )
    test_loader = data.DataLoader(
        test_sets,
        cfg.model.training_settings.test_batch_size, 
        shuffle=False, 
        drop_last=True,
        num_workers=cfg.model.training_settings.num_workers
    )

    #--------------------------------------------------------------------------------------------

    model = hydra.utils.instantiate(
        cfg.model.use_model
    ).to(device)

    #--------------------------------------------------------------------------------------------
    criterion = torch.nn.CrossEntropyLoss(
        reduction = 'mean',
        weight = torch.tensor(cfg.dataset.rate)
    ).to(device)
    

    optimizer = hydra.utils.instantiate(
        cfg.model.optimizer,
        params = model.parameters()
    )

    scheduler = hydra.utils.instantiate(
        cfg.model.scheduler,
        optimizer = optimizer
    )
    #--------------------------------------------------------------------------------------------

    input1 = torch.randn(1,cfg.dataset.hsi_channel , *cfg.dataset.img_size).to(device)
    input2 = torch.randn(1,cfg.dataset.lidar_channel, *cfg.dataset.img_size).to(device)


    flops,params=profile(model,inputs=(input1,input2))
    print("flops: {:.3f} G , params: {:.3f} M ".format(flops/1e9,params/1e6))
    torch.cuda.empty_cache()

    #--------------------------------------------------------------------------------------------
    
    #load pt-------------------
    #model.load_state_dict(torch.load(arg.co_meta,weights_only=True),False)


    for epoch in range(cfg.model.training_settings.epochs):
        #------------------------------------------
        model.train()
        loop = tqdm(enumerate(train_loader),total = len(train_loader),leave=False)
        for batch_idx, (hsi, lidar , label , mask) in loop :
            hsi = hsi.to(device)
            lidar = lidar.to(device)
            label = (label).to(device)
            mask = mask.to(device)
            pred = model(hsi , lidar)
            optimizer.zero_grad()
    
            pred = pred.view(pred.shape[0], cfg.dataset.num_classes, -1)
            label = label.view(label.shape[0], -1)
            mask = mask.view(mask.shape[0], -1)

            idx = torch.where(mask > 0)
            pred = pred[idx[0], :, idx[1]]
            label = label[idx]
            
            loss = criterion(pred, label) +model.loss

            loss.backward()
            optimizer.step()
            loop.set_description('train')
        scheduler.step()
        train_loss = loss.item()

        #------------------------------------------
        model.eval()
        with torch.no_grad():
            cm = 0
            test_loss = 0

            loop = tqdm(enumerate(test_loader),total =len(test_loader),leave=False)
            for batch_idx, (hsi , lidar, label , mask) in loop:
                hsi = hsi.to(device)
                lidar = lidar.to(device)
                label = (label).to(device)
                mask = mask.to(device)
                pred = model(hsi,lidar)

                pred = pred.view(pred.shape[0], cfg.dataset.num_classes, -1)
                label = label.view(label.shape[0], -1)
                mask = mask.view(mask.shape[0], -1)

                idx = torch.where(mask > 0)
                pred = pred[idx[0], :, idx[1]]
                label = label[idx]

                loss = criterion(pred, label) +model.loss
            
                test_loss += loss.item()

                pred = pred.detach().argmax(dim=-1)
                cm += utils.metric.compute_confusion_matrix(pred,label,cfg.dataset.num_classes)
                loop.set_description('eval')
        
        oa,aa,k = utils.metric.calculate_metrics(cm)
        torch.cuda.empty_cache()

        test_loss = test_loss/(batch_idx+1)
        print("[ epoch: {}  ,  train_loss: {:.4f}  ,  test_loss: {:.4f}  ".format(epoch,train_loss,test_loss), '//' ,
           " OA: {:.2f} %  ,  AA: {:.2f} %  ,  k: {:.2f} %  ]".format( oa*100,aa*100,k*100))

        #check point

        if isinstance(model,torch.nn.DataParallel):
            torch.save(model.module.state_dict(),cfg.model.training_settings.checkpoint)
        else :
            torch.save(model.state_dict(),cfg.model.training_settings.checkpoint)
        if epoch == cfg.model.training_settings.early_stop:
            break



if __name__ == "__main__" :
    main()

    