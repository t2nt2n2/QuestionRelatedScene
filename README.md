# QuestionRelatedScene
classification the scene related by question

## Overview

- Linux 16.04
- NVIDIA GeForce GTX 1080Ti
- pytorch 1.2
- Python3 (>= 3.5.2)
- CUDA 10.0
- Cudnn 7.5
- tqdm (Optional for test.py)
- tensorboard >= 1.14 (see [Tensorboard Visualization][#tensorboardx-visualization])

## install
install all dependency
I'll provide docker image later

## How to use

title: question based related scene proposal

input: s02ep05 에피소드의 각 scene들과 question

output: 각 scene들과 question과의 distance(0~1)

#### 1. import the docker file
```
yonsei_VTT.tar docker import
```

#### 2. unzip pytorch-template-master.zip

#### 3. python infer.py -r saved/models/Friends_Bert/0831_172003/model_best.pth -d 0 명령어로 원하는 question과 scene에 대해 실행(question과 scene은 infer.py에 ID와 question 변수로 설정 가능), -d 는 GPU 넘버
```
python infer.py -r saved/models/Friends_Bert/0831_172003/model_best.pth -d 0
```

#### 4 output.json에 각 scene과의 유사도 순위가 나오게 됩니다.

## caution

#### 1. 테스트 타임이 많이 걸리는데 모든 에피소드에 대해서 데이터를 처리하기 때문. 실제 데모에서 할때는 미리 처리해놓은 데이터로 돌리면 실행시간은 단축 됨

#### 2. pytorch 1.2 cuda 10.0 으로 작업을 진행. 다른 도커 이미지로 실행시 pip install 로는 pandas, tensorboardX가 필요

#### 3. training과 validation은 s02ep05를 제외한 나머지로 진행. 테스트는 s02ep05에 존재하는 Question으로 진행


## V. Related Works
1. https://github.com/huggingface/transformers
2. https://github.com/victoresque/pytorch-template

