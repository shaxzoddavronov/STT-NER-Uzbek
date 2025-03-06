## Model and Dataset Links:
* STT: https://huggingface.co/ShakhzoDavronov/whisper-large-lora-uz
* NER: https://huggingface.co/ShakhzoDavronov/xlm-roberta-lora-ner-uz
* STT Dataset: https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0/viewer/uz
* NER Dataset: https://huggingface.co/datasets/risqaliyevds/uzbek_ner

## Overview
This project focuses on developing a Speech-to-Text (STT) system specifically for the Uzbek language and enhancing it with Named Entity Recognition (NER) capabilities. The system converts spoken Uzbek into text and then processes the transcribed output to identify and extract meaningful entities such as names, locations, organizations, and other relevant information.

## NER Details
The large version of XLM-RoBERTa model include over 570 million paramters which cause time inefficency and computing more resources (GPU/CPU). So We utilized one of the PEFT (Parameter Efficent Fine-Tuning) technique  called LoRA (Low Rank Adaption) used to fine-tune large neural networks efficiently by training only a small subset of parameters while keeping the majority of the model frozen. We trained only ~ 2 % paramters (12 millon) of original model.

![image alt](https://github.com/shaxzoddavronov/STT-NER-Uzbek/blob/main/images/NER_trainable_params.png?raw=true)

### Base Model:
* https://huggingface.co/FacebookAI/xlm-roberta-large
  
### Dataset:
* https://huggingface.co/datasets/risqaliyevds/uzbek_ner
  
### Fine-Tuned Model Hyperparameters:
* batch size = 8
* learning rate = 2e-4
* weight_decay = 0.01
* trained epochs = 3
* fp16 = True
* evaluation and save strategy = 'epoch'
  
### LoRA Configuration:
* rank = 128
* alpha = 32
* dropout = 0.01

### Model Result:
![image alt](https://github.com/shaxzoddavronov/STT-NER-Uzbek/blob/main/images/NER%20Result.png?raw=true)

### Testing
Tested with sample sentence:

![image alt](https://github.com/shaxzoddavronov/STT-NER-Uzbek/blob/main/images/NER%20Testing.png?raw=true)

### Training Porcess Code:
xxxxxxxxx


## STT Details
Whisper model developed by OpenAI used to fine-tune on STT task. Large version 2 of the model chosen to train with LoRa adapter like XLM-RoBERTa. Since, it has over 1.5B paramaters and it is almost impossible to train on  open source cloud-based Jupyter notebook environments (kaggle, google colab). We trained only ~ 1 % paramters (15 millon) of original model.

![image alt](https://github.com/shaxzoddavronov/STT-NER-Uzbek/blob/main/images/STT_trainable_params.png?raw=true)

### Base Model:
* https://huggingface.co/openai/whisper-large-v2
  
### Dataset:
* https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0/viewer/uz

### Entites:
B-GPE (Countries)
B-LOC (Location)
B-PERSON (Person)
B-ORG (Organization)
B-DATE (Date)
B-EVENT (Events)

### Fine-Tuned Model Hyperparameters:
* batch size = 8
* learning rate = 1e-3
* gradient accumulation = 1
* trained epochs = 1
* fp16 = True
* evaluation and save strategy = 'steps'
* logging stpes = 300
  
### LoRA Configuratio:
* rank = 32
* alpha = 64
* dropout = 0.05

### Model Result:
![image alt](https://github.com/shaxzoddavronov/STT-NER-Uzbek/blob/main/images/STT%20Loss.png?raw=true)
![image alt](https://github.com/shaxzoddavronov/STT-NER-Uzbek/blob/main/images/STT%20Result.png?raw=true)

### Testing
Tested with sample sentence:

![image alt](https://github.com/shaxzoddavronov/STT-NER-Uzbek/blob/main/images/STT%20Testing.png?raw=true)

### Training Process Code:
xxxxxxxxxxx

## Usage Guidance
#### Requirements
##### Python>3.10
##### Packages: Transformers, Datasets, Torch, Evaluate and others.

``` python
pip install -r requirements.txt
```

### Testing STT model:
``` python
!pip install -U bitsandbytes
```

``` python
import torch
from transformers import AutomaticSpeechRecognitionPipeline
from transformers import WhisperTokenizer,WhisperForConditionalGeneration,WhisperProcessor
from peft import PeftModel, PeftConfig

stt_model_id = "ShakhzoDavronov/whisper-large-lora-uz"
language = "Uzbek"
task = "transcribe"
stt_config = PeftConfig.from_pretrained(stt_model_id)
stt_model = WhisperForConditionalGeneration.from_pretrained(
    stt_config.base_model_name_or_path, load_in_8bit=True, device_map="auto"
)

stt_model = PeftModel.from_pretrained(stt_model, stt_model_id)
stt_tokenizer = WhisperTokenizer.from_pretrained(stt_config.base_model_name_or_path, language=language, task=task)
stt_processor = WhisperProcessor.from_pretrained(stt_config.base_model_name_or_path, language=language, task=task)
stt_feature_extractor = stt_processor.feature_extractor
forced_decoder_ids = stt_processor.get_decoder_prompt_ids(language=language, task=task)
stt_pipe = AutomaticSpeechRecognitionPipeline(model=stt_model, tokenizer=stt_tokenizer, feature_extractor=stt_feature_extractor)


def transcribe(audio):
    with torch.cuda.amp.autocast():
        text = stt_pipe(audio, generate_kwargs={"forced_decoder_ids": forced_decoder_ids}, max_new_tokens=255)["text"]
    return text

text=transcribe(audio)
print(text)
```

### Testing NER model:
```python
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import pipeline
from peft import PeftModel, PeftConfig

label_names=['O','B-PERSON','B-GPE','B-ORG','B-LOC','B-DATE','B-EVENT']
num_labels=len(label_names)
id_to_label = {i: label for i, label in enumerate(label_names)}
label_to_id = {label:i for i, label in enumerate(label_names)}

ner_model_id = "ShakhzoDavronov/xlm-roberta-lora-ner-uz"
ner_config = PeftConfig.from_pretrained(ner_model_id)
ner_model = AutoModelForTokenClassification.from_pretrained(ner_config.base_model_name_or_path, num_labels=len(label_names),
                                                        id2label=id_to_label, label2id=label_to_id)

ner_tokenizer = AutoTokenizer.from_pretrained(ner_config.base_model_name_or_path)
ner_model = PeftModel.from_pretrained(ner_model,ner_model_id)

ner_pipe=pipeline('ner',model=ner_model,tokenizer=ner_tokenizer)

text="Toshkentda Shavkat Mirziyoyev Rossiya prezidentini kutib oldi"
ner=ner_pipe(text)
for entity in ner:
    print(entity)
```

### Pipeline

```python
def final_pipeline(audio,ner_pipe,stt_pipe):

  extracted_text=stt_pipe(audio, generate_kwargs={"forced_decoder_ids": forced_decoder_ids}, max_new_tokens=255)["text"]
  ner_entities=ner_pipe(extracted_text)
  print(extracted_text)
  for entity in ner_entities:
    print(entity)

final_pipeline('Test_1.wav',ner_pipe,stt_pipe)
```

#### If you have any question, contact me:
* Me: Shakhzod Davronov
* Telegram : https://t.me/shaxzod_067
* LinkedIn:  https://www.linkedin.com/in/shaxzod-davronov/
* Email: shaxzoddavronov0@gmail.com
