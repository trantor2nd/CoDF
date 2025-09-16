import os
import logging
import torch
from torch.utils import data
from thop import profile
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

import model
import utils
import utils.func
import utils.LULC_loader
import utils.metric

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg : DictConfig) -> None:

    logger.debug(OmegaConf.to_yaml(cfg))
    save_path = cfg.model.train.save_dir + cfg.model.name +'/'+ cfg.dataset.name
    foler = os.path.exists(save_path)
    if not foler:
        os.makedirs(save_path)
    seed = cfg.model.train.seed
    device = cfg.model.train.device
    num_workers = cfg.model.train.num_workers
    batch_size = cfg.model.train.batch_size
    epochs = cfg.model.train.epochs

    #--------------------------------------------------------------------------------------------
    utils.func.set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available())
    #--------------------------------------------------------------------------------------------
    
    train_sets = utils.LULC_loader.LULC_dataset(cfg,phase ='train')
    test_sets = utils.LULC_loader.LULC_dataset(cfg,phase ='test')

    logger.debug("train : {} , test : {}".format(len(train_sets),len(test_sets)))

    train_loader = data.DataLoader(
        train_sets,
        batch_size, 
        shuffle=False, 
        drop_last=True,
        num_workers=num_workers
    )
    test_loader = data.DataLoader(
        test_sets,
        batch_size, 
        shuffle=False, 
        drop_last=True,
        num_workers=num_workers
    )

    #--------------------------------------------------------------------------------------------

    model = hydra.utils.instantiate(
        cfg.model.use_model
    ).to(device)

    #--------------------------------------------------------------------------------------------
    criterion = torch.nn.CrossEntropyLoss(
        reduction = 'mean',
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
    logger.debug("flops: {:.3f} G , params: {:.3f} M ".format(flops/1e9,params/1e6))
    torch.cuda.empty_cache()
    

    #training------------------------------------------------------------------------------------
    best_oa = 0
    for epoch in range(epochs):
        logger.debug(f'Epoch: {epoch} start-------')
        
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
            
            loss = criterion(pred, label) + model.loss

            loss.backward()
            optimizer.step()
            loop.set_description('train')
        
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

                loss = criterion(pred, label) + model.loss
            
                test_loss += loss.item()

                pred = pred.detach().argmax(dim=-1)
                cm += utils.metric.compute_confusion_matrix(pred,label,cfg.dataset.num_classes)
                loop.set_description('eval')
        
        oa,aa,k = utils.metric.calculate_metrics(cm)
        torch.cuda.empty_cache()

        test_loss = test_loss/(batch_idx+1)
      
        logger.debug("Epoch: {}  |  train_loss: {:.4f}  |  test_loss: {:.4f}  ||  OA: {:.2f} %  |  AA: {:.2f} %  |  k: {:.2f} %".format(epoch,train_loss,test_loss, oa*100,aa*100,k*100))

        #check point
        if(oa > best_oa):
            best_oa = oa
            best_aa = aa
            best_k = k
            best_model_path = save_path + '/best_model.pt'
            checkpoint = model.module.state_dict() if isinstance(model,torch.nn.DataParallel) else model.state_dict()
            torch.save(checkpoint, best_model_path)
        scheduler.step()

    logger.debug("Best: OA: {:.2f} %  |  AA: {:.2f} %  |  k: {:.2f} %".format( best_oa*100,best_aa*100,best_k*100))
    logger.debug(f'best model saved at {best_model_path}')

if __name__ == "__main__" :
    main()

    