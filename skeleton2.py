
    
def logistic(x):
    return softPlusP(x)
    
def logisticP(x):
    return logistic(x)*(1- logistic(x))    
    
def softPlusP(x):
    return 1/(1 + np.exp(-x) )

def tanhP(x):    
    return 1 - np.tanh(x)**2    

def RMSProp(smoothGrad, currGrad):
    rho = 0.9
    smoothGrad = np.sqrt( rho*np.power(smoothGrad, 2)  + (1 - rho)*np.power(currGrad, 2) )
    return smoothGrad

def getSample(D, sampleID, k):
    
    n = len(sampleID)
    sIndex = [i for i in range(n) if sampleID[i] == k]
    rIndex = [i for i in range(n) if sampleID[i] != k]
    cvPartition = namedtuple('data', 'R E')
    
    cvData = namedtuple('data','X Y labels')
    if D.labels is None:
        
        split = cvPartition(cvData(D.X[rIndex,:], D.Y[rIndex,:], None ), cvData(D.X[sIndex,:], D.Y[sIndex,:], None ))
    else:
        split = cvPartition(cvData(D.X[rIndex,:], D.Y[rIndex,:], [D.labels[i] for i in rIndex]), 
                            cvData(D.X[sIndex,:], D.Y[sIndex,:],[D.labels[i] for i in sIndex]) )
    return split    

z0 = ActivationFunction(rlu, rluP)   
z1 = ActivationFunction(identity, unit)   
z2= ActivationFunction(logistic, logisticP)
Z3 = ActivationFunction(np.tanh, tanhP)

seed = 0
random.seed(seed)
np.random.seed(seed)
'''  START computing ... '''    


if 1 ==0:
    path = '/home/brian/NeuralNets/Data/breast-cancer-wisconsin.txt'
    D = breastCancerData(path)
    fns = [z1.evaluate, z2.evaluate, z2.evaluate]
    dfns = [z1.differentiate, z2.differentiate, z2.differentiate]
    aConstant = .01

if 1 ==0:
    path = '/home/brian/NeuralNets/Data/bostonHousingData.txt'
    D = BostonHousing(path)
    aConstant = .01
    
if 1 ==1:
    path = '/home/brian/NeuralNets/Data/parkinsons_updrs.csv'
    D = parkinsonsData(path)
    
    fns = [z0.evaluate, z3.evaluate, z1.evaluate]
    dfns = [z0.differentiate, z3.differentiate, z1.differentiate]

n, p = dim(D.X)
n, s = dim(D.Y)
print(n,p, s)

g = [p, p, s]

if len(g) > len(fns) +1  :
    print('Mismatch between number of layers and number of activation functions')
    sys.exit()
else:
    print(g)    
    
if D.labels is None:           
    objFn = rmse
    dEdyhatf = dEdyhatSqr 
else: 
    objFn = crossEntropy
    dEdyhatf = dEdyhatCE

K = 10
cvDict = dict.fromkeys(range(K))
sampleID = [random.choice(range(K)) for i in range(n)]

for k in range(K):
    try:
        del progress
    except(NameError):
        pass
    
    F = getSample(D, sampleID, k)        
    X = F.R.X
    Y = F.R.Y
    
    if D.labels is None:
        beta = np.linalg.solve(X.T*X, X.T*Y) 
        E = F.E.Y - F.E.X*beta
        rsq = rSqr(F.E.Y, E)    
        
        print('\nN train = ',dim(F.R.Y)[0], '\tn test = ',dim(F.E.Y)[0] ,
                 '\tLinear regression adj R2 (test) = ',rsq)
    else:        
        print('\nN train = ',dim(F.R.Y)[0], '\tn test = ',dim(F.E.Y)[0])

    initialList = initialize(g, F.R.X, fns, dfns)
    yHat, xList, hList, gList, zpList, vList, iList, sgList = initialList  
        
    #  Begin iterations 
    for it in range(100): 
        # Forward Propagation 
        xList, zpList, yHat = fProp(xList, hList, fns, dfns, zpList)
        dEdyhat = dEdyhatf(F.R.Y , yHat) 
                                  
        # Evaluate progress 
        obsAcc = testAcc(F.E, hList, fns)
        objFunction = objFn(Y, yHat)
        obsAcc.append(objFunction)
        try:
            progress = [.9*progress[i] + .1*obsAcc[i] for i in range(len(progress))]
        except(NameError):
            progress = obsAcc
            
        string = '\r'+str(k) + '/'+str(K)+' \t'+str(it)
        if D.labels is None:
            for i in range(s):
                string += '\t r-sqr = '+ str(round(progress[i],3)) 
            string +='\t obj = '+ str(round(progress[len(progress)-1],5))
        else:
            string += '\t acc = '+ str(round(progress[0],3)) \
                    + '\t obj = '+ str(round(progress[len(progress)-1],5))
        
        print(string+'\t a = '+str(round(a,5)),end="")
    
    
    #print('Completed ', k+1, 'of ', K)  
    cvDict[k] = [dim(F.E.Y)[0], obsAcc[:s]]   
    
    #cvDict[k] = [dim(F.E.Y)[0], rsq]   
cvAcc = [sum([cvDict[k][0] * cvDict[k][1][i] for k in range(K)])/n for i in range(s)]    
print('\nCV adjusted R-sqr = '+'  '+'  '.join([str(round(a,3)) for a in cvAcc]))

sys.exit()
