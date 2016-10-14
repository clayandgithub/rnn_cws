#!/usr/bin/python
# -*- coding: utf-8 -*-

#2016年 01月 28日 星期四 17:14:03 CST by Demobin

def _print(hiddenstates, V):
    s = "    " + " ".join(("%7d" % i) for i in range(len(V))) + "\n"
    for i, state in enumerate(hiddenstates):
        s += "%.5s: " % state
        s += " ".join("%.7s" % ("%f" % v[i]) for v in V)
        s += "\n"
    print(s)

#标准viterbi算法，参数为观察状态、隐藏状态、概率三元组(初始概率、转移概率、观察概率)
def viterbi(obs, states, start_p, trans_p, emit_p):

    lenObs = len(obs)
    lenStates = len(states)

    V = [[0.0 for col in range(lenStates)] for row in range(lenObs)]
    path = [[0 for col in range(lenObs)] for row in range(lenStates)]

    #t = 0时刻
    for y in range(lenStates):
        #V[0][y] = start_p[y] * emit_p[y][obs[0]]
        V[0][y] = start_p[y] * emit_p[y][0]
        path[y][0] = y

    #t > 1时
    for t in range(1, lenObs):
        newpath = [[0.0 for col in range(lenObs)] for row in range(lenStates)]

        for y in range(lenStates):
            prob = -1
            state = 0
            for y0 in range(lenStates):
                #nprob = V[t - 1][y0] * trans_p[y0][y] * emit_p[y][obs[t]]
                nprob = V[t - 1][y0] * trans_p[y0][y] * emit_p[y][t]
                if nprob > prob:
                    prob = nprob
                    state = y0
                    #记录最大概率
                    V[t][y] = prob
                    #记录路径
                    newpath[y][:t] = path[state][:t]
                    newpath[y][t] = y

        path = newpath

    prob = -1
    state = 0
    for y in range(lenStates):
        if V[lenObs - 1][y] > prob:
            prob = V[lenObs - 1][y]
            state = y

    #_print(states, V)
    return prob, path[state]

def example():
    #隐藏状态
    hiddenstates = ('Healthy', 'Fever')
    #观察状态
    observations = ('normal', 'cold', 'dizzy')

    #初始概率
    '''
    Healthy': 0.6, 'Fever': 0.4
    '''
    start_p = [0.6, 0.4]
    #转移概率
    '''
    Healthy' : {'Healthy': 0.7, 'Fever': 0.3},
    Fever' : {'Healthy': 0.4, 'Fever': 0.6}
    '''
    trans_p = [[0.7, 0.3], [0.4, 0.6]]
    #发射概率/输出概率/观察概率
    '''
    Healthy' : {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
    Fever' : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6}
    '''
    emit_p = [[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]]

    return viterbi(observations,
                   hiddenstates,
                   start_p,
                   trans_p,
                   emit_p)

if __name__ == '__main__':
    print(example())