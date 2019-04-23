import numpy as np
from sklearn import datasets
import pandas as pd

data_breast_cancer=datasets.load_breast_cancer()
data=data_breast_cancer.data
target=data_breast_cancer.target
data_frame=np.insert(data,data.shape[1],target,axis=1)
df_breast_cancer=pd.DataFrame(data_frame,columns=list(data_breast_cancer.feature_names)+['target'])

def entropy(target_column):
    
    values,occurences=np.unique(target_column,return_counts=True)
    return sum([-occ/float(sum(occurences))*np.log2(occ/float(sum(occurences))) for occ in occurences])

def InformationGain_discrete(data,variable,target):
    
    All_entropy=entropy(data[target])
    values,occurences=np.unique(data[variable],return_counts=True)
    entropy_to_be_retrieved=sum([occurences[i]/float(sum(occurences))*entropy(data[data[variable]==values[i]][target]) for i in range(len(values))])
    
    return All_entropy - entropy_to_be_retrieved

def InformationGain_continuous(data,variable,target):
    
    All_entropy=entropy(data[target])
    sorted_values=np.sort(data[variable])
    list_entropies=[]
    
    for value in sorted_values[1:-1]:
        occurences_inf=np.unique(data[data[variable]<=value],return_counts=True)[1].tolist()
        occurences_sup=np.unique(data[data[variable]>value],return_counts=True)[1].tolist()
        occurences=[sum(occurences_inf),sum(occurences_sup)]
        list_entropies.append(sum([-occ/float(sum(occurences))*np.log2(occ/float(sum(occurences))) for occ in occurences]))
     
    index_min=np.argmin(list_entropies)
    
    #Output a tuple: the information gain and the value leading to it
    return All_entropy - list_entropies[index_min], sorted_values[index_min+1]

def attribute_to_split(data,variables,target):
    
    information_gains=list(map(lambda variable:InformationGain(data,variable,target),variables))
    feature_index=np.argmax(information_gains)
    variable_name=variables[feature_index]
    
    return data[variable_name], variable_name

class TreeNode:
    def __init__(self):
        self.value=None
        self.index=None
        self.children=None
        
    def add_child(self, node):
        assert isinstance(node, TreeNode)
        self.children.append(node)

Tree=TreeNode()

def build_tree(data_to_split,data,variables,target,tree,additional_memory_for_label,minimum_per_row=10):
    if len(np.unique(data_to_split[target]))==1:
        return np.unique(data_to_split[target])[0]
    
    if len(data_to_split)<=minimum_per_row or len(variables)==0 :
        #continue wiith putting the maximum labelof the original data
