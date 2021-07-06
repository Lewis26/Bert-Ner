# Bert-Ner
# 观点挖掘整理

#### 

#### 四元组

- AspectTerm:评论原文中的商品属性特征词。
  例如“价格很便宜”中的“价格”。该字段结果须与评论原文中的表述保持一致。
- OpinionTerm:评论原文中，用户对商品某一属性所持有的观点。
  例如“价格很便宜”中的“很便宜”。该字段结果须与评论原文中的表述保持一致。
- Category:用户对某一属性特征的观点所蕴含的情感极性，即负面、中性或正面三类。
- Polarity:相似或同类的属性特征词构成的属性种类。
  例如“快递”和“物流”两个属性特征词都可归入“物流”这一属性种类。

#### 评分标准

1、相同ID内逐一匹配各四元组，若AspectTerm，OpinionTerm，Category，Polarity四个字段均正确，则该四元组正确；
2、预测的四元组总个数记为P；真实标注的四元组总个数记为G；正确的四元组个数记为S：
（1）精确率： Precision=S/P
（2）召回率： Recall=S/G
（3）F值:F1-score=(2*Precision*Recall)/(Precision+Recall)

#### 标注模式

使用BIO数据标注模式，总共标注两个实体：`AspectTerms`,`OpinionTerms`
编码方式为："B-ASP", "I-ASP", "B-OPI", "I-OPI"
句子之间用空行分隔。

解决联合标注问题的最简单的方法，就是将其转化为原始标注问题。标准做法就是使用BIO标注。

BIO标注：将每个元素标注为“B-X”、“I-X”或者“O”。其中，“B-X”表示此元素所在的片段属于X类型并且此元素在此片段的开头，“I-X”表示此元素所在的片段属于X类型并且此元素在此片段的中间位置，“O”表示不属于任何类型。

训练数据如下

```txt
,id,AspectTerms,A_start,A_end,OpinionTerms,O_start,O_end,Categories,Polarities,text
0,1,_, , ,很好,0,2,整体,正面,很好，超值，很好用
```

标注后数据如下

```txt
很 B-OPI
好 I-OPI
， O
超 B-OPI
值 I-OPI
， O
很 B-OPI
好 I-OPI
用 I-OPI
```



#### 模型训练

##### 输入数据格式

输入为[定常向量](https://blog.csdn.net/u011984148/article/details/99921480)。

##### BERT简介

[BERT](https://links.jianshu.com/go?to=https%3A%2F%2Farxiv.org%2Fpdf%2F1810.04805.pdf)是2018年google 提出来的预训练的语言模型，并且它打破很多NLP领域的任务记录，其提出在nlp的领域具有重要意义。预训练的(pre-train)的语言模型通过无监督的学习掌握了很多自然语言的一些语法或者语义知识，之后在做下游的nlp任务时就会显得比较容易。**BERT在做下游的有监督nlp任务时就像一个做了充足预习的学生去上课，那效果肯定事半功倍。**之前的word2vec，glove等Word Embedding技术也是通过无监督的训练让模型预先掌握了一些基础的语言知识，但是word embeding技术无论从预训练的模型复杂度(可以理解成学习的能力)，以及无监督学习的任务难度都无法和BERT相比。可以参考[BERT实战](https://blog.csdn.net/w417950004/article/details/99080193)。

##### 训练方式

BERT 为了让模型能够比较好的掌握自然语言方面的知识，提出了下面两种预训练的任务：

1. 遮盖词的预测任务（mask word prediction），如下图所示：
   将输入文本中15%的token随机遮盖，然后输入给模型，最终希望模型能够输出遮盖的词是什么，这就是让模型在做**完形填空**。
2. 下一个句子预测任务如下图所示：给模型输入A，B两个句子，让模型判断B句子是否是A句子的下一句。这个任务是希望模型能够学到句子间的关系，更近一步的加强模型对自然语言的理解。

##### FLAGS

利用该函数，可以实现在命令行中设置需要设定的参数来运行程序。这样的话，就可以不用在源代码中指定参数，相当于在命令行中传递需要设定的参数。

定义下边文件名称为`test_flags.py`

```python
from absl import flags
from absl import app
 
FLAGS = flags.FLAGS
 
#1、第一个是参数名称，第二个参数是默认值，第三个是参数描述
flags.DEFINE_string('model', None, 'model to run')
 
def main(argv):
    print('Hello World')
    print('selected model', FLAGS.model)
 
if __name__ == '__main__':  
    app.run(main)  #2、执行main函数
```

运行上面的文件

```python
# 1、运行示例程序
python test_flags.py
 
# 2、更改相应参数
python test_flags.py --model "My model"
 
# 3、获得帮助信息
python test_flags.py -help
python test_flags.py -helpfull
```

##### Tf模型格式

TensorFlow提供了两种模型格式：

1. checkpoints：这种格式依赖于创建模型的代码。
2. SavedModel：这种格式与创建模型的代码无关。

Checkpoints文件是这样的一个二进制文件，好比是一个中转站，Tensorflow针对这一需提供了Saver类把变量名映射到对应的tensor值，并可以从checkpoints文件中恢复变量。

```python

```



