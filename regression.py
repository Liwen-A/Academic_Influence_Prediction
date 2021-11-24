from numpy.core.numeric import NaN
from sklearn.kernel_ridge import KernelRidge
from sklearn import metrics
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt




def main():
    # load csv into data
    df = pd.read_csv('./Physicsdata.csv') # load dataset into pandas datastructure
    #df = df.append(pd.read_csv('./Biologydata.csv'))
    #df = df.append(pd.read_csv('./Chemistrydata.csv'))
    #df = df.append(pd.read_csv('./Civil Engineeringdata.csv'))
    #df = df.append(pd.read_csv('./Computer sciencedata.csv'))
    #df = df.append(pd.read_csv('./Economicsdata.csv'))
    #df = df.append(pd.read_csv('./Material Sciencedata.csv'))
    #df = df.append(pd.read_csv('./Mathematicsdata.csv'))
    #df = df.append(pd.read_csv('./Medicinedata.csv'))
    #df = df.append(pd.read_csv('./Physicsdata.csv'))

    #nan_value = float("NaN")
    #df.replace("", nan_value, inplace=True)
    #df.dropna(subset = ['fieldsOfStudy'], inplace=True)
    
    # clean up data with 0 values for reference count, which means missing info
    df['referenceCount'].replace(0, np.nan, inplace=True)
    df.dropna(subset=['referenceCount'], inplace=True)
    # remove blank fields of study
    df['fieldsOfStudy'].replace("", np.nan, inplace=True)
    df.dropna(subset=['fieldsOfStudy'], inplace=True)
    df.drop_duplicates(subset=['paperId'])
    n = df.shape[0] # number of training examples
    
    # shuffle rows
    df = df.sample(frac=1)

    vn = int(n * 0.6)
    tn = int(n * 0.8)
    # separate all feature vectors
    year = df['year'] # feature 1
    refcount = df['referenceCount'] # feature 2
    
    citcount = df['citationCount'] # label 
    icitcount = df['influentialCitationCount'] # feature 3
    
    field = df['fieldsOfStudy']
    categories = pd.Categorical(field)
    encoded = categories.codes # encoded categories into numbers / feature 4
    hist = df.plot.hist(bins=5)

    # convert all pandas to numpy
    #yeararray = year.to_numpy().reshape(-1,1)
    #refarray = refcount.to_numpy().reshape(-1,1)
    citarray = citcount.to_numpy().reshape(-1,1)
    #icitarray = icitcount.to_numpy().reshape(-1,1)
    #catarray = encoded.to_numpy().reshape(-1,1)
    

    # 2d matrix of features
    farray = np.column_stack((year.to_numpy(), refcount.to_numpy(), icitcount.to_numpy(), encoded))
    #print(np.shape(farray))
    #print(citcount.to_numpy().reshape(-1,1))
    # metric{“linear”, “additive_chi2”, “chi2”, “poly”, “polynomial”, 
    #  “rbf”, “laplacian”, “sigmoid”, “cosine”}
    a = np.logspace(-5, 0, 6)
    k = 'cosine'
    pscore = -1e10
    for i in range(len(a)):
        krr = KernelRidge(alpha=a[i], kernel=k)
        krr.fit(farray[:vn], citarray[:vn])
        print(krr.score(farray[vn:tn], citarray[vn:tn]))
        if krr.score(farray[vn:tn], citarray[vn:tn]) > pscore:
            pscore = krr.score(farray[vn:tn], citarray[vn:tn])
            besta = a[i]
    krr = KernelRidge(alpha=besta, kernel=k)
    krr.fit(farray[:vn], citarray[:vn])
    print(krr.score(farray[tn:], citarray[tn:]))
    print(besta)
    '''
    krr = KernelRidge(alpha=a, kernel='poly')
    krr.fit(farray[:tn], citarray[:tn])
    print(krr.score(farray[tn:], citarray[tn:]))
    krr = KernelRidge(alpha=a, kernel='rbf')
    krr.fit(farray[:tn], citarray[:tn])
    print(krr.score(farray[tn:], citarray[tn:]))
    krr = KernelRidge(alpha=a, kernel='laplacian')
    krr.fit(farray[:tn], citarray[:tn])
    print(krr.score(farray[tn:], citarray[tn:]))
    krr = KernelRidge(alpha=a, kernel='sigmoid')
    krr.fit(farray[:tn], citarray[:tn])
    print(krr.score(farray[tn:], citarray[tn:]))
    krr = KernelRidge(alpha=a, kernel='cosine')
    krr.fit(farray[:tn], citarray[:tn])
    print(krr.score(farray[tn:], citarray[tn:]))
    '''
    
    #blank = blank.astype('object')
    #print(list(filter(None,re.split("','|']|\['", field[0]))))
    
    #for i in range(4000):
        
        #blank[i] = list(filter(None,re.split("','|']|\['", field[i]))) # converts to only categories
    
    # convert field into a vector of numbers
    # assign each index of a hot vector to a field.
    # [math, physics, chemistry, computer science, aeronautics, material science, civil engineering, biology,
    #  medicine, sociology, economics]
    #for i in range(n):
    #    hotfield = 

    


    # parse data correctly into feature vectors


if __name__ == '__main__':
    main()

