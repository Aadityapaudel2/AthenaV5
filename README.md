# AthenaV5
Olympiad Level Math Performance Local Model Shell


For finetuning, 
first use prepare_data.py  go to `runtraining.ps1`, specify your address inside project root through
example:
```
    [string]$ModelPath = "models/Qwen3-0.6B-Base",
    [string]$TrainFile = "Finetune/trainingdata/samples/athena_commandments_train.jsonl",
    [string]$OutputDir = "Finetune/trainingdata/output/run1"
```

Store your models inside models folder. 

For GUI, 
Find line 19 : `DEFAULT_FINE_TUNED = ATHENA_V5_ROOT / "models" / "Qwen3-4B"`, and change the model currently set as Qwen3-4B. 






    
