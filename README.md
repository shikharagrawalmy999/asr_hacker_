## Training  
1. Download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/)  
2. Make `preprocessed` folder in the LJSpeech directory(setup the directory structure as given in `prepare_data.py`) and do preprocessing of the data using `prepare_data.py`  
3. Set the `data_path` in `hparams.py` to the `preprocessed` folder  
4. Train your own BVAE-TTS model  
```python
python train.py --gpu=0 --logdir=baseline  
```  
## Inference
Load the pretrained waveglow model as :
```waveglow_path = 'training_log/waveglow_256channels_ljs_v2.pt'
waveglow = torch.load(waveglow_path, map_location=torch.device("cpu"))['model']```

Load your trained BVAE-TTS model as :
```checkpoint_path = f"training_log/baseline/checkpoint_11000"
model = Model(hp)
model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu"))['state_dict'])```

Run inference.py