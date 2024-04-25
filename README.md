# MA-AGIQA：Large Multi-modality Model Assiste AI-Generated Image Quality Assessment

## Semantic Feature Extraction
We use official mPLUG-Owl2 to extract semantic features. The feature extraction codes are based on [Q-Align](https://github.com/Q-Future/Q-Align), great thanks to them!
First, down load their code
```shell 
git clone https://github.com/Q-Future/Q-Align.git
cd Q-Align
```
then 
- put json files containing information of training data to "Q-Align/playground/data/test_jsons" like "Q-Align/playground/data/test_jsons/AGIQA_3k.json".
- put getFeature.py like "Q-Align/q_align/evaluate/getFeature.py".

You can find them under "q_align" file in this repository and get semantic feature by
```shell 
python "q_align/evaluate/getFeature.py"
```
if you have error when connect to Hugging Face, we recommand you use
```shell 
HF_ENDPOINT=https://hf-mirror.com python "q_align/evaluate/getFeature.py"
```
## Train and Test
Download and transfer to MA-AGIQA. If you've download this repository, just implement the "cd" code.
```shell 
git clone https://github.com/wangpuyi/MA-AGIQA.git
cd MA-AGIQA
```
Train and Test
```shell 
python train.py
```
