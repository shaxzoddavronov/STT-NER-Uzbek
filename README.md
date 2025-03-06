# Speech-to-Text (STT) and Named Entity Recognition (NER) Integrated Pipeline for Uzbek

## Model and Dataset Links:
* STT: https://huggingface.co/ShakhzoDavronov/whisper-large-lora-uz
* NER: https://huggingface.co/ShakhzoDavronov/xlm-roberta-lora-ner-uz
* STT Dataset: https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0/viewer/uz
* NER Dataset: https://huggingface.co/datasets/risqaliyevds/uzbek_ner

## Overview
This project focuses on developing a Speech-to-Text (STT) system specifically for the Uzbek language and enhancing it with Named Entity Recognition (NER) capabilities. The system converts spoken Uzbek into text and then processes the transcribed output to identify and extract meaningful entities such as names, locations, organizations, and other relevant information.

## NER Details
The large version of XLM-RoBERTa model include over 570 million paramters which cause time inefficency and computing more resources (GPU/CPU). So We utilized one of the PEFT (Parameter Efficent Fine-Tuning) technique  called LoRa (Low Rank Adaption) used to fine-tune large neural networks efficiently by training only a small subset of parameters while keeping the majority of the model frozen. We trained only ~ 2 % paramters (12 millon) of original model.

![Alt text]([images/my-image.png](https://github.com/shaxzoddavronov/STT-NER-Uzbek/blob/main/images/NER_trainable_params.png))
