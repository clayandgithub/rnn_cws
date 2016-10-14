# -*- coding: utf-8 -*-

'''
python pretrain.py input_file cws_info_filePath cws_data_filePath
'''

import json
import h5py
import string
import codecs
import sys
import time

corpus_tags = ['S', 'B', 'M', 'E']
retain_unknown = 'retain-unknown'
retain_padding = 'retain-padding'

def saveCwsInfo(path, cwsInfo):
    '''保存分词训练数据字典和概率'''
    print('save cws info to %s'%path)
    fd = open(path, 'w')
    (initProb, tranProb), (vocab, indexVocab) = cwsInfo
    j = json.dumps((initProb, tranProb))
    fd.write(j + '\n')
    for char in vocab:
        fd.write(char.encode('utf-8') + '\t' + str(vocab[char]) + '\n')
    fd.close()

def loadCwsInfo(path):
    '''载入分词训练数据字典和概率'''
    print('load cws info from %s'%path)
    fd = open(path, 'r')
    line = fd.readline()
    j = json.loads(line.strip())
    initProb, tranProb = j[0], j[1]
    lines = fd.readlines()
    fd.close()
    vocab = {}
    indexVocab = [0 for i in range(len(lines))]
    for line in lines:
        rst = line.strip().split('\t')
        if len(rst) < 2: continue
        char, index = rst[0].decode('utf-8'), int(rst[1])
        vocab[char] = index
        indexVocab[index] = char
    return (initProb, tranProb), (vocab, indexVocab)

def saveCwsData(path, cwsData):
    '''保存分词训练输入样本'''
    print('save cws data to %s'%path)
    #采用hdf5保存大矩阵效率最高
    fd = h5py.File(path,'w')
    (X, y) = cwsData
    fd.create_dataset('X', data = X)
    fd.create_dataset('y', data = y)
    fd.close()

def loadCwsData(path):
    '''载入分词训练输入样本'''
    print('load cws data from %s'%path)
    fd = h5py.File(path,'r')
    X = fd['X'][:]
    y = fd['y'][:]
    fd.close()
    return (X, y)

def sent2vec2(sent, vocab, ctxWindows = 5):

    charVec = []
    for char in sent:
        if char in vocab:
            charVec.append(vocab[char])
        else:
            charVec.append(vocab[retain_unknown])
    #首尾padding
    num = len(charVec)
    pad = int((ctxWindows - 1)/2)
    for i in range(pad):
        charVec.insert(0, vocab[retain_padding] )
        charVec.append(vocab[retain_padding] )
    X = []
    for i in range(num):
        X.append(charVec[i:i + ctxWindows])
    return X

def sent2vec(sent, vocab, ctxWindows = 5):
    chars = []
    for char in sent:
        chars.append(char)
    return sent2vec2(chars, vocab, ctxWindows = ctxWindows)

def doc2vec(fname, vocab):
    '''文档转向量'''

    #一次性读入文件，注意内存
    fd = codecs.open(fname, 'r', 'utf-8')
    lines = fd.readlines()
    fd.close()

    #样本集
    X = []
    y = []

    #标注统计信息
    tagSize = len(corpus_tags)
    tagCnt = [0 for i in range(tagSize)]
    tagTranCnt = [[0 for i in range(tagSize)] for j in range(tagSize)]

    #遍历行
    for line in lines:
        #按空格分割
        words = line.strip('\n').split()
        #每行的分词信息
        chars = []
        tags = []
        #遍历词
        for word in words:
            #包含两个字及以上的词
            if len(word) > 1:
                #词的首字
                chars.append(word[0])
                tags.append(corpus_tags.index('B'))
                #词中间的字
                for char in word[1:(len(word) - 1)]:
                    chars.append(char)
                    tags.append(corpus_tags.index('M'))
                #词的尾字
                chars.append(word[-1])
                tags.append(corpus_tags.index('E'))
            #单字词
            else: 
                chars.append(word)
                tags.append(corpus_tags.index('S'))

        #字向量表示
        lineVecX = sent2vec2(chars, vocab, ctxWindows = 7)

        #统计标注信息
        lineVecY = []
        lastTag = -1
        for tag in tags:
            #向量
            lineVecY.append(tag)
            #lineVecY.append(corpus_tags[tag])
            #统计tag频次
            tagCnt[tag] += 1
            #统计tag转移频次
            if lastTag != -1:
                tagTranCnt[lastTag][tag] += 1
            #暂存上一次的tag
            lastTag = tag

        X.extend(lineVecX)
        y.extend(lineVecY)

    #字总频次
    charCnt = sum(tagCnt)
    #转移总频次
    tranCnt = sum([sum(tag) for tag in tagTranCnt])
    #tag初始概率
    initProb = []
    for i in range(tagSize):
        initProb.append(tagCnt[i]/float(charCnt))
    #tag转移概率
    tranProb = []
    for i in range(tagSize):
        p = []
        for j in range(tagSize):
            p.append(tagTranCnt[i][j]/float(tranCnt))
        tranProb.append(p)

    return X, y, initProb, tranProb

def genVocab(fname, delimiters = [' ', '\n']):

    #一次性读入文件，注意内存
    fd = codecs.open(fname, 'r', 'utf-8')
    data = fd.read()
    fd.close()

    vocab = {}
    indexVocab = []
    #遍历
    index = 0
    for char in data:
        #如果为分隔符则无需加入字典
        if char not in delimiters and char not in vocab:
            vocab[char] = index
            indexVocab.append(char)
            index += 1

    #加入未登陆新词和填充词
    vocab[retain_unknown] = len(vocab)
    vocab[retain_padding] = len(vocab)
    indexVocab.append(retain_unknown)
    indexVocab.append(retain_padding)
    #返回字典与索引
    return vocab, indexVocab

def load(fname):
    print 'train from file', fname
    delims = [' ', '\n']
    vocab, indexVocab = genVocab(fname)
    X, y, initProb, tranProb = doc2vec(fname, vocab)
    print len(X), len(y), len(vocab), len(indexVocab)
    print initProb
    print tranProb
    return (X, y), (initProb, tranProb), (vocab, indexVocab)

if __name__ == '__main__':
    start_time = time.time()

    if len(sys.argv) < 4:
        print globals()['__doc__'] % locals()
        sys.exit(1)
    input_file, cws_info_filePath, cws_data_filePath = sys.argv[1:4]

    (X, y), (initProb, tranProb), (vocab, indexVocab) = load(input_file)
    saveCwsInfo(cws_info_filePath, ((initProb, tranProb), (vocab, indexVocab)))
    saveCwsData(cws_data_filePath, (X, y))

    end_time = time.time()
    print("used time : %d s" % (end_time - start_time))