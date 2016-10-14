说明：
利用rnn实现中文分词算法
源码参考:http://www.jianshu.com/p/7e233ef57cb6
数据集下载地址：http://pan.baidu.com/s/1jIyNT7w

训练步骤：
1 用现有的语料库（已经切分好）训练出word2vec的model
2 预训练处理语料库得到训练输入和测试输入
3 构建rnn并进行训练，在训练的同时测试准确率
4 根据训练好的model得到可能的序列组合，并利用viterbi算法选择出其中可能性最大的一个序列