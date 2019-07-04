# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:25:48 2019

@author: HYEJEONG

"""
from mathematics import *
import sequence as seq
import statistics as stat

import pandas as pd
from math import log #Problem with float --> change to log 
  
class Hmm(object):
    """
    Main Class of Hiddem Markov Model. 
    Hiddem Markov Model(lambda) is specified with three parameters
    Pi : prob_start, start probability
    A : prob_trans, transition probability
    B : prob_emit, emission probability
    State list and symbol list are also included.   
    """
    def __init__(self, prob_start, prob_trans, prob_emit, statelist, symbollist):
        #Define model with five parameters
        self._prob_start = prob_start
        self._prob_trans = prob_trans
        self._prob_emit = prob_emit
        self._statelist = list(statelist)
        self._symbollist = list(symbollist)

    def get(self):
        #Show parameters
        return self._prob_start, self._prob_trans, self._prob_emit, self._statelist, self._symbollist
        
    def zero(self):   
        #Initialize to all probabilites to 0.0
        self._prob_trans = {}
        self._prob_emit = {}
        self._prob_start = {}

        for state in self._statelist:
            self._prob_trans[state] = {}
            self._prob_emit[state] = {}
            for post_state in self._statelist:
                self.prob_trans[state][post_state] = 0.0
            for symbol in self._symbollist:
                self._prob_emit[state][symbol] = 0.0
            self._prob_start[state] = 0.0
  
    def check(self, sequence):
        #Check the sequence is appropriate
        if sequence not in self._symbollist:
            print('Not available sequence')
 
    def prob_start(self):
        #Show start probability
        if self._prob_start == None:    
            return 0
        return self._prob_start
 
    def prob_trans(self):
        #Show transition probability
        if self._prob_trans == None:    
            return 0   
        return self._prob_trans    
    
    def prob_emit(self):
        #Show emission probability
        if self._prob_emit == None:    
            return 0      
        return self._prob_emit
    
    def statelist(self):
        #Show state list 
        if self._statelist == None:    
            return None     
        return self._statelist
    
    def symbollist(self):
        #Show symbol list 
        if self._symbollist == None:    
            return None     
        return self._symbollist    
            
    def emupdate(self, sequence):  #For updating, Baum-Welch(Forward-Backward algorithm)
        """
        This is a method for updating A, B by calculating value with A*B 
        It is based on Expectation-Maximization algorithm
        """
        a = self.forward(sequence)
        b = self.backward(sequence)
        
        # Expectation value (E-step)
        g = [] # get gamma 
        g_smooth = []
        sum_prob = 0.0
        for t in range(0, len(sequence)): # 0, 1, ... l-1 step
            g.append({})
            sum_prob = 0.0
            for state in self._statelist:
                g[t][state] = a[t][state] * b[t][state]
                sum_prob += g[t][state]    

            g_smooth.append(sum_prob)
            if sum_prob > 1e-300:
                for state in self._statelist:
                    g[t][state] /= sum_prob

        x = [] # get xi
        x_smooth = []
        for t in range(0, len(sequence)-1): # 0, 1, ... l-2 step
            x.append({})
            sum_prob = 0.0
            for pre_state in self._statelist:            
                x[t][pre_state] = {}
                for post_state in self._statelist:
                    x[t][pre_state][post_state] = a[t][pre_state]  \
                        * self._prob_trans[pre_state][post_state] \
                        * self._prob_emit[post_state][sequence[t+1]] \
                        * b[t+1][post_state] 
                    sum_prob += x[t][pre_state][post_state]    
                x_smooth.append(sum_prob)
            
            if sum_prob > 1e-300:
                for pre_state in self._statelist:
                    for post_state in self._statelist:
                        x[t][pre_state][post_state] /= sum_prob 


        #Maximization step (M-step)
        #Laplace smoothing 
        p = 0.00001
        for state in self._statelist:
            #for start probabilities
            self._prob_start[state] = (g[0][state] + p) / (1 + p * len(self._statelist))   

            #for transition probabilities   
            for post_xstate in self._statelist:   
                sum_sum_prob = 0.0 
                sum_x = 0.0
                for t in range(0, len(sequence)-1): #fraction denominator
                    sum_prob= 0.0
                    for post_state in self._statelist:                
                        sum_prob += x[t][state][post_state]     
                    sum_sum_prob += sum_prob 

                    sum_x += x[t][state][post_xstate] #fraction numerator
                   
                    self._prob_trans[state][post_xstate] = (sum_x + p) \
                    / (sum_sum_prob + p*len(self._statelist))

            #for emission probabilities
            sum_g = 0.0
            for t in range(0, len(sequence)):         
                sum_g += g[t][state]
            sum_g += g[len(sequence)-1][state]

            sum_g_emit = {}
            for symbol in self._symbollist:
                sum_g_emit[symbol] = 0.0
                
            for t in range(0, len(sequence)):
                sum_g_emit[sequence[t]] += g[t][state] 
            
            for symbol in self._symbollist:
                self._prob_emit[state][symbol] = (sum_g_emit[symbol] + p) \
                / (sum_g + p*len(self._statelist))  
        return self._prob_emit, self._prob_trans
                
    def forward(self, sequence):
        """
        This is a method for calculating forward probability. 
        output must be a list as below. 
        a = [ {'h':... , 'e':... , '_':... },    a_{0} 
                            ...    
              {'h':... , 'e':... , '_':... }, ]  a_{length-1}
        """    
        a = []   # forward probability, alpha (list for [t steps]; 
                 # dict for probabilities of states {'h', 'e', '_'})            
                 # alpha is saved for next alpha, dynamics programming
        c = [{}] # scaling factor for very very small number   
        
        for t in range(0, len(sequence), +1): # 0, 1, ... , l-1 step               
            a.append({})
            if t == 0: 
                for state in self._statelist:                 
                    
                    a[t][state] = self._prob_start[state] * self._prob_emit[state][sequence[0]]                 
            else:
                for state in self._statelist:
                    
                    sum_prob = 0.0
                    for pre_state in self._statelist:                     
                        sum_prob += a[t-1][pre_state] * self._prob_trans[pre_state][state]
                    a[t][state] = sum_prob * self._prob_emit[state][sequence[t-1]]

            if sum(list(a[t] .values())) > 1e-300:
                c = 1 / sum(list(a[t].values()))
                for state in self._statelist:
                    a[t][state] *= c 

        return a 

    def backward(self, sequence):     
        """
        This is a method for calculating backward probability. 
        Contrary, b is calculated from the last step, so b_{0} is calculated and stacked further.
        Output must be a list as below.
        b = [ {'h':... , 'e':... , '_':... },    b_{0}
                            ...    
              {'h':... , 'e':... , '_':... }, ]  b_{length-1}
        """   
        b = [] # backward probability, beta (list for [t states];
               # dict for probabilities of states {'h', 'e', '_'})            
               # beta is saved for next beta, dynamics programming
        c = [] # scaling factor for very very very small number   
        for t in range(len(sequence)-1, -1, -1): # 0, 1, ... , l-1 step               
            b.insert(0, {})       
            for state in self._statelist: 
                if t == (len(sequence)-1):
                    b[0][state] = 1.0        
                else:      
                    sum_prob = 0.0
                    for post_state in self._statelist:                     
                        sum_prob += b[1][post_state] * self._prob_trans[state][post_state]

                    b[0][state] = sum_prob * self._prob_emit[state][sequence[t]]

            if sum(list(b[0].values())) > 1e-300:
                c = 1 / sum(list(b[0].values()))
                for state in self._statelist:
                    b[0][state] *= c 
        return b    
    
    def viterbi(self, sequence): 
        """ 
        This is a method for Viterbi algorithm to decode.  
        Output must be a list as below .            
        v = [ {'h':... , 'e':... , '_':... },    v_{0}
                            ...    
              {'h':... , 'e':... , '_':... }, ]  v_{length-1} 
        """ 
        v = []
        dec_state = []
        dec_prob = []
        for t in range(0, len(sequence)): # state number : 0, 1, ... , l-1 step                    
            v.append({})
            
            if t == 0: 
                for state in self._statelist:                 
                    v[t][state] = self._prob_start[state] * self._prob_emit[state][sequence[0]] 
            else:                
                for state in self._statelist:       
                    v[t][state] = vmax[0] * self._prob_trans[vmax[1]][state] \
                    * self._prob_emit[state][sequence[t-1]]

            vmax = list(max(zip(v[t].values(), v[t].keys())))            
            
            if vmax[0] > 1e-300 :
                c = 1 / sum(list(v[t].values()))
                for state in self._statelist:
                   v[t][state] *= c 
                vmax[0] *= c 
                
            dec_prob.append(vmax[0])           
            dec_state.append(vmax[1])        

        return [v[t], dec_state, vmax[0]]

    def initialize(self, proteinset): 
        #Initialize a model by parameters from outside of class. 
        #This is because we can not used the model parameter by simple counting method after training 
        self._prob_start = initialset(proteinset)[1][0]
        self._prob_trans = initialset(proteinset)[1][1]
        self._prob_emit = initialset(proteinset)[1][2]
        self._statelist = initialset(proteinset)[2][0]
        self._symbollist = initialset(proteinset)[2][1]   
        return prob_start, prob_trans, prob_emit, statelist, symbollist
    
    def train(self, initialset, trainset, initial=1, iteration = False, d=0.0101, maxiter=100): 
        """
        This is a method for updating parameter by EM algorithm, related to stop iteration.
        Stopping criteria follows the log-likelihood.
        Parameters   |   Description
                     |
        initial      | Initializing method/dataset. Default = 1(set that already modeled) 
        iteration    | Stopping criteria. Default = False(maxiter is not used.)
        d            | Stopping criteria for convergence. Default = 0.0101. 
        maxiter      | Stopping criteria for iteration. Default = 100. 
        """
        if initial == 0:
            self.zero()
        elif initial == 1:
            pass    
        elif initial == 2:
            self.initialize(trainset)
        elif initial ==3 :
            self.initialize(initialset)
            
        proteins = trainset[0]
        structures = trainset[1]

        pre_prob = 0.0
        for protein in proteins:
            prob = self.viterbi(protein)
            pre_prob += log(prob[2]) #expectation
        pre_prob /= len(proteins)
        print(pre_prob/len(proteins))
        
        for i in range(maxiter): 
            post_prob = 0.0
            print('Iteration number is ... ', i+1)
            for protein in proteins:
                #E-M update transition probabilites and emission probabilities 
                self.emupdate(protein)
                prob = self.viterbi(protein) #viterbi probability 
                post_prob += log(prob[2])
            post_prob /= len(proteins)
            print(pre_prob, post_prob)
            if iteration == False:
                if (post_prob > pre_prob) and (abs(post_prob - pre_prob) < d):
                    break
            #if post_prob <= pre_prob:   
            pre_prob = post_prob 
    
        return i, pre_prob/len(proteins), post_prob/len(proteins)

    def decode(self, proteinset):
        """
        This is a method for decoding(translating) by Viterbi algorithm.
        To find the  sequence that has highest probability. 
        """
        proteins = proteinset[0]
        structures = proteinset[1]
        
        decode = []
        likelihood = []
        
        for protein, structure in zip(proteins, structures):
            viterbi = self.viterbi(protein)
            decode.append(viterbi[1])
            likelihood.append(viterbi[2])

        decodeset = [proteins, decode]
        return decodeset          

def initialset(proteinset):
    """
    This is a method for finding parameters by raw data sets. 
    Raw data sets are seperated as below. 
     
    """
    proteins = proteinset[0] 
    structures = proteinset[1]
    
    symbolcount = {}
    statecount = {}

    startcount = {}
    transcount = {}
    statesymbolcount = {} 
    
    for protein, structure in zip(proteins, structures):
        pre_state = {}
        count1d(startcount, structure[0])    
        for symbol, state in zip(protein, structure): #single protein
            count1d(statecount, state) 
            count1d(symbolcount, symbol)
            count2d(statesymbolcount, state, symbol) # for transmission probability           
            if pre_state != {}: 
                count2d(transcount, pre_state, state) # for emittance probability            
            pre_state = state

    statelist = list(statecount.keys())
    symbollist = list(symbolcount.keys()) 

    prob_trans = norm2d(transcount, statelist, statelist)
    prob_emit = norm2d(statesymbolcount, statelist, symbollist)
    prob_start = norm1d(startcount, statelist)
    #print(prob_start)
    return [startcount, transcount, statesymbolcount], [prob_start, prob_trans, prob_emit], \
           [statelist, symbollist]

def initialHmm(proteinset):      
    prob_start = initialset(proteinset)[1][0]
    prob_trans = initialset(proteinset)[1][1]
    prob_emit = initialset(proteinset)[1][2]
    statelist = initialset(proteinset)[2][0]
    symbollist = initialset(proteinset)[2][1]
    return Hmm(prob_start, prob_trans, prob_emit, statelist, symbollist )
"""
def varname(p): #https://stackoverflow.com/questions/592746/how-can-you-print-a-variable-name-in-python
  for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
    m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
    if m:
      return m.group(1)
"""
def printdata(file, data):
    print(data)
    with open(file, "w") as f:
        f.write("\n\n######result of decoded sequence##### \n ")
       # f.write(varname(data))  
        f.write("\n############################################# \n")
        f.write("\n")
        f.write(str(data))
        f.write("\n")
        size = list(map(len, data[0]))
        for i in range(len(data[0])):
            f.write(str(i+1))
            f.write(str(data[0][i]))
            f.write("\n")
            f.write(str(data[1][i]))
            f.write("\n\n\n")            
        pd.set_option('display.max_colwidth', 3*max(size)) 
        pd.set_option('display.max_rows', 3*max(size)) 
        tr = pd.DataFrame(data).T
        f.write(tr.to_string())

if __name__ == "__main__":
    path_train = '.\dataset\protein-secondary-structure.train'
    path_test = '.\dataset\protein-secondary-structure.test'
    path_decode = '.\output\raw_data.csv'
    rawlines = None
    while True:
        try:
            path_train = input('Insert input file for training \n(Exit key : Ctrl + C) \n(Example set : .\dataset\protein-secondary-structure.train) : \n ')
            rawlines = seq.lineseq(path_train)
            #print(rawlines)
            if rawlines == None:
                print('Empty input file. Try again')
                continue
            else:
                break
        except FileNotFoundError or NameError as err:  
            print(err)
            print('This is not available input file. Try again.')
            continue

    rawlines = None
    
    while True:
        try:
            path_test = input('Insert input file for testing \n(for example: .\dataset\protein-secondary-structure.test) : \n ')
            rawlines = seq.lineseq(path_test)
            #print(rawlines)
            if rawlines == None:
                print('Empty input file. Try again.')
                continue
            else:
                break

        except FileNotFoundError or NameError as err:  
            print(err)
            print('This is not available input file. Select options with integer number below (1, 2, 3).', \
                '1. Try another file', \
                '2. Use example input file (.\dataset\protein-secondary-structure.test)', \
                '3. Exit : Ctrl + C', \
                sep='\n')  
            trytype = input()
            print('Option selected :', trytype)
            if trytype == str(1): 
                continue
            elif trytype == str(2):
                path_test = '.\dataset\protein-secondary-structure.test'
                break
            else:
                print('Incorrect value. Try again')
                continue     

    print(path_train, 'is a file for training\n')
    print(path_test, 'is a file for testing\n\n')
    print('decoded raw data is saved as')
    print('.\output\raw_data_method.csv')
    
    trainset = seq.getproteinset(path_train)
    testset = seq.getproteinset(path_test)

    trainmodel = initialHmm(trainset)
    data_simple = trainmodel.decode(testset)
    printdata(r'.\output\raw_data_simpleHMM.csv', data_simple)
    
    trainmodel_conv = initialHmm(trainset)
    trainmodel_conv.train(trainset, trainset)
    data_conv = trainmodel_conv.decode(testset)
    printdata(r'.\output\raw_data_EMHMMconv.csv', data_conv)
    
    trainmodel_iter = initialHmm(trainset)
    trainmodel_iter.train(trainset, trainset, iteration=True)
    data_iter = trainmodel_iter.decode(testset)
    printdata(r'.\output\raw_data_EMHMMiter.csv', data_iter)
    
    """
    Want more statistical analysis?
    printdata('summary5.csv', stat.summary(data_simple))
    printdata('summary4.csv', stat.summary(data_conv))
    printdata('summary4.csv', stat.summary(data_iter))

    printdata('accuracy5.csv', stat.accuracy(testset, data_simple))
    printdata('accuracy5.csv', stat.accuracy(testset, data_conv))
    printdata('accuracy5.csv', stat.accuracy(testset, data_iter))
    """
    """
    for curiosity, this is start from zero probability for all parameters, (lambda, A, B)

    with open("Result_comparetrains.csv", "a") as f:
        trainmodel.train(trainset, trainset, initial=0)
        f.write("\n\n######EM trained result - Convergence / from initial \n\n")
        f.write(str(trainmodel.decode(testset)))
        f.write("\n\n")
    T = pd.DataFrame(trainmodel.decode(testset)).T
    T.to_csv("Result_comparetrains.csv", mode='a', header=True)  # Save in result.txt file

    with open("Result_comparetrains.csv", "a") as f:
        trainmodel.train(trainset, trainset, initial=0, iteration=True)
        f.write("\n\n######EM trained result - maxiter / from initial \n\n")
        f.write(trainmodel.decode(testset))
        f.write("\n\n")
    T = pd.DataFrame(trainmodel.decode(testset)).T
    T.to_csv("Result_comparetrains.csv", mode='a', header=True)  # Save in result.txt file
    """