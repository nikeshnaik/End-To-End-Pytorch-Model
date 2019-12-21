# End-To-End-Pytorch-Model

End to End Pytorch Modelling from custom dataloader,save&load checkpoints, hyperparatmeters tuning file and deployment using flask API.

To train model using hyperparameters and configuration
```
python train.py -c config.json
```
To Test model 
```
python test.py -c config.json -r saved/models/*/*/model_best.pth
```
Deploy using Flask Api
```
python api.py
```
Inspired by https://github.com/victoresque/pytorch-template

