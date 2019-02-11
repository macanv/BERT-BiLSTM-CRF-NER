# BERT-BiLSMT-CRF-NER
Tensorflow solution of NER task Using BiLSTM-CRF model with Google BERT Fine-tuning

使用谷歌的BERT模型在BLSTM-CRF模型上进行预训练用于中文命名实体识别的Tensorflow代码'

中文文档请查看https://blog.csdn.net/macanv/article/details/85684284  如果对您有帮助，麻烦点个star,谢谢~~  

Welcome to star this repository!

The Chinese training data($PATH/NERdata/) come from:https://github.com/zjy-ucas/ChineseNER 
  
The CoNLL-2003 data($PATH/NERdata/ori/) come from:https://github.com/kyzhouhzau/BERT-NER 
  
The evaluation codes come from:https://github.com/guillaumegenthial/tf_metrics/blob/master/tf_metrics/__init__.py  


Try to implement NER work based on google's BERT code and BiLSTM-CRF network!
This project may be more close to process Chinese data. but other language only need Modify a small amount of code.

THIS PROJECT ONLY SUPPORT Python3.  
###################################################################
## Download project and install  
You can install this project by:  
```
pip install bert-base==0.0.3 -i https://pypi.python.org/simple
```
OR
```angular2html
git clone https://github.com/macanv/BERT-BiLSTM-CRF-NER
cd BERT-BiLSTM-CRF-NER/
python3 setup.py install
```

## UPDATE:
1. fix Missing loss error
2. add label_list params in train process, so you can using -label_list xxx to special labels in training process.  

    
## Train model:
You can use -help to view the relevant parameters of the training named entity recognition model, where data_dir, bert_config_file, output_dir, init_checkpoint, vocab_file must be specified.
```angular2html
bert-base-ner-train -help
```
![](./pictures/ner_help.png)  
  

train/dev/test dataset is like this:
```
海 O
钓 O
比 O
赛 O
地 O
点 O
在 O
厦 B-LOC
门 I-LOC
与 O
金 B-LOC
门 I-LOC
之 O
间 O
的 O
海 O
域 O
。 O
```
The first one of each line is a token, the second is token's label, and the line is divided by a blank line. The maximum length of each sentence is [max_seq_length] params.  
You can get training data from above two git repos  
You can training ner model by running below command:  
```angular2html
bert-base-ner-train \
    -data_dir {your dataset dir}\
    -output_dir {training output dir}\
    -init_checkpoint {Google BERT model dir}\
    -bert_config_file {bert_config.json under the Google BERT model dir} \
    -vocab_file {vocab.txt under the Google BERT model dir}
```
you can special labels using -label_list params, the project get labels from training data.  
```angular2html
# using , split
-labels 'B-LOC, I-LOC ...'
OR save label in a file like labels.txt, one line one label
-labels labels.txt
```    

After training model, the NER model will be saved in {output_dir} which you special above cmd line.  

## As Service
Many server and client code comes from excellent open source projects: [bert as service of hanxiao](https://github.com/hanxiao/bert-as-service) If my code violates any license agreement, please let me know and I will correct it the first time.
and NER server/client service code can be applied to other tasks with simple modifications, such as text categorization, which I will provide later.
Welcome to submit your request, if you want to share it on Github or my work.  

You can use -help to view the relevant parameters of the NER as Service:
which ner_model_dir, bert_model_dir is need
```
bert-base-serving-start -help
```
![](./pictures/server_help.png)

and than you can using below cmd start ner service:
```angular2html
bert-base-serving-start \
    -ner_model_dir C:\workspace\python\BERT_Base\output\ner2 \
    -bert_model_dir F:\chinese_L-12_H-768_A-12
    -model_pb_dir C:\workspace\python\BERT_Base\model_pb_dir
```
as you see:   
mode: If mode is NER, then the service identified by the named entity will be started. If it is BERT, it will be the same as the [bert as service] project.  
bert_model_dir: bert_model_dir is a BERT model, you can download from https://github.com/google-research/bert
ner_model_dir: your ner model checkpoint dir
model_pb_dir: model freeze save dir, after run optimize func, there will contains like ner_model.pb binary file  
>You can download my ner model from：https://pan.baidu.com/s/1m9VcueQ5gF-TJc00sFD88w, ex_code: guqq  
Set ner_mode.pb to model_pb_dir, and set other file to ner_model_dir  

You can see below service starting info:
![](./pictures/service_1.png)
![](./pictures/service_2.png)


you can using below code test client:  
```angular2html
import time
from bert_base.client import BertClient

with BertClient(show_server_config=False, check_version=False, check_length=False, mode='NER') as bc:
    start_t = time.perf_counter()
    str = '1月24日，新华社对外发布了中央对雄安新区的指导意见，洋洋洒洒1.2万多字，17次提到北京，4次提到天津，信息量很大，其实也回答了人们关心的很多问题。'
    rst = bc.encode([str, str])
    print('rst:', rst)
    print(time.perf_counter() - start_t)
```
you can see this after run the above code:
![](./pictures/server_ner_rst.png)

  
# The following tutorial is an old version and will be removed in the future.

## How to train
#### 1. Download BERT chinese model :  
 ```
 wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip  
 ```
#### 2. create output dir
create output path in project path:
```angular2html
mkdir output
```
#### 3. Train model

##### first method 
```
  python3 bert_lstm_ner.py   \
                  --task_name="NER"  \ 
                  --do_train=True   \
                  --do_eval=True   \
                  --do_predict=True
                  --data_dir=NERdata   \
                  --vocab_file=checkpoint/vocab.txt  \ 
                  --bert_config_file=checkpoint/bert_config.json \  
                  --init_checkpoint=checkpoint/bert_model.ckpt   \
                  --max_seq_length=128   \
                  --train_batch_size=32   \
                  --learning_rate=2e-5   \
                  --num_train_epochs=3.0   \
                  --output_dir=./output/result_dir/ 
 ```       
 ##### OR replace the BERT path and project path in bert_lstm_ner.py
 ```
 if os.name == 'nt': #windows path config
    bert_path = '{your BERT model path}'
    root_path = '{project path}'
else: # linux path config
    bert_path = '{your BERT model path}'
    root_path = '{project path}'
 ```
 Than Run:
 ```angular2html
python3 bert_lstm_ner.py
```

### USING BLSTM-CRF OR ONLY CRF FOR DECODE!
Just alter bert_lstm_ner.py line of 450, the params of the function of add_blstm_crf_layer: crf_only=True or False  

ONLY CRF output layer:
```
    blstm_crf = BLSTM_CRF(embedded_chars=embedding, hidden_unit=FLAGS.lstm_size, cell_type=FLAGS.cell, num_layers=FLAGS.num_layers,
                          dropout_rate=FLAGS.droupout_rate, initializers=initializers, num_labels=num_labels,
                          seq_length=max_seq_length, labels=labels, lengths=lengths, is_training=is_training)
    rst = blstm_crf.add_blstm_crf_layer(crf_only=True)
```
  
  
BiLSTM with CRF output layer
```
    blstm_crf = BLSTM_CRF(embedded_chars=embedding, hidden_unit=FLAGS.lstm_size, cell_type=FLAGS.cell, num_layers=FLAGS.num_layers,
                          dropout_rate=FLAGS.droupout_rate, initializers=initializers, num_labels=num_labels,
                          seq_length=max_seq_length, labels=labels, lengths=lengths, is_training=is_training)
    rst = blstm_crf.add_blstm_crf_layer(crf_only=False)
```

## Result:
all params using default
#### In dev data set:
![](./pictures/picture1.png)

#### In test data set
![](./pictures/picture2.png)

#### entity leval result:
last two result are label level result, the entitly level result in code of line 796-798,this result will be output in predict process.
show my entity level result :
![](./pictures/03E18A6A9C16082CF22A9E8837F7E35F.png)
> my model can download from baidu cloud:  
>链接：https://pan.baidu.com/s/1GfDFleCcTv5393ufBYdgqQ 提取码：4cus  
NOTE: My model is trained by crf_only params

## ONLINE PREDICT
If model is train finished, just run
```angular2html
python3 terminal_predict.py
```
![](./pictures/predict.png)
 
 ## Using NER as Service

#### Service 
Using NER as Service is simple, you just need to run the python script below in the project root path:
```angular2html
python3 runs.py \ 
    -mode NER
    -bert_model_dir /home/macan/ml/data/chinese_L-12_H-768_A-12 \
    -ner_model_dir /home/macan/ml/data/bert_ner \
    -model_pd_dir /home/macan/ml/workspace/BERT_Base/output/predict_optimizer \
    -num_worker 8
```

  
You can download my ner model from：https://pan.baidu.com/s/1m9VcueQ5gF-TJc00sFD88w, ex_code: guqq  
Set ner_mode.pb to model_pd_dir, and set other file to ner_model_dir and than run last cmd  
![](./pictures/service_1.png)
![](./pictures/service_2.png)


#### Client
The client using methods can reference client_test.py script
```angular2html
import time
from client.client import BertClient

ner_model_dir = 'C:\workspace\python\BERT_Base\output\predict_ner'
with BertClient( ner_model_dir=ner_model_dir, show_server_config=False, check_version=False, check_length=False, mode='NER') as bc:
    start_t = time.perf_counter()
    str = '1月24日，新华社对外发布了中央对雄安新区的指导意见，洋洋洒洒1.2万多字，17次提到北京，4次提到天津，信息量很大，其实也回答了人们关心的很多问题。'
    rst = bc.encode([str])
    print('rst:', rst)
    print(time.perf_counter() - start_t)
```
NOTE: input format you can sometime reference bert as service project.    
Welcome to provide more client language code like java or others.  
 ## Using yourself data to train
 if you want to use yourself data to train ner model,you just modify  the get_labes func.
 ```angular2html
def get_labels(self):
        return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
```
NOTE: "X", “[CLS]”, “[SEP]” These three are necessary, you just replace your data label to this return list.  
Or you can use last code lets the program automatically get the label from training data
```angular2html
def get_labels(self):
        # 通过读取train文件获取标签的方法会出现一定的风险。
        if os.path.exists(os.path.join(FLAGS.output_dir, 'label_list.pkl')):
            with codecs.open(os.path.join(FLAGS.output_dir, 'label_list.pkl'), 'rb') as rf:
                self.labels = pickle.load(rf)
        else:
            if len(self.labels) > 0:
                self.labels = self.labels.union(set(["X", "[CLS]", "[SEP]"]))
                with codecs.open(os.path.join(FLAGS.output_dir, 'label_list.pkl'), 'wb') as rf:
                    pickle.dump(self.labels, rf)
            else:
                self.labels = ["O", 'B-TIM', 'I-TIM', "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
        return self.labels

```


## NEW UPDATE
2019.1.30 Support pip install and command line control  

2019.1.30 Add Service/Client for NER process  

2019.1.9: Add code to remove the adam related parameters in the model, and reduce the size of the model file from 1.3GB to 400MB.  
  
2019.1.3: Add online predict code  



## reference: 
+ The evaluation codes come from:https://github.com/guillaumegenthial/tf_metrics/blob/master/tf_metrics/__init__.py

+ [https://github.com/google-research/bert](https://github.com/google-research/bert)
      
+ [https://github.com/kyzhouhzau/BERT-NER](https://github.com/kyzhouhzau/BERT-NER)

+ [https://github.com/zjy-ucas/ChineseNER](https://github.com/zjy-ucas/ChineseNER)

+ [https://github.com/hanxiao/bert-as-service](https://github.com/hanxiao/bert-as-service)
> Any problem please open issue OR email me(ma_cancan@163.com)
