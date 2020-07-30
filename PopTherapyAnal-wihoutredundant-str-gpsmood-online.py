
# coding: utf-8

# # Analysis on Pop therapy Data
# 
# ## - Application of Random Forest UCB

# In[1]:


#Dependencies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.tree import export_graphviz
#from subprocess import call
#import os
#from sklearn import tree
#from IPython import display
#from sklearn.ensemble import RandomForestClassifier
#from sklearn import model_selection
#import math
#import seaborn as sns


# ### 0. Construct class for Random Forest UCB

# In[2]:


##RF UCB algorithm

from sklearn.ensemble import RandomForestRegressor

class RF_UCB:
    def __init__(self,v,reference):
        self.v=v #constant that controls degree of eploration
        self.RF=RandomForestRegressor(n_estimators=1000, min_samples_leaf=1, max_features=0.33, bootstrap=True)
        self.Y=[]
        self.X=[]
        self.reference=reference #table that contains binary indicators of the 8 interventions
    def select_ac(self,x,t): #function for selecting intervention given input covariate x
        
        # make xfull matrix using x. The 8 rows share the same x but have different intervention indicators.
        xfull=pd.concat([x for i in range(8)], axis=1).T
        xfull.index=self.reference.index
        xfull=pd.concat([xfull,self.reference],axis=1)
            
        if t==0:
            ind=np.random.choice(np.arange(8)) #if there is no accumulated data, select intervention randomly with equal probability
        else:
            predictions=self.RF.predict(xfull) #estimate the reward of each of the 8 interventions based on current RF model
            stds=[self.comp_std(self.RF,xfull.iloc[i]) for i in range(8)] #compute standard deviation based on current RF model
            ucbs=np.array(predictions)+self.v*np.array(stds) #compute UCB = estimate + v*std
            ind=np.random.choice(np.where(ucbs==ucbs.max())[0]) #choose the intervention with maximum ucb. select randomly if there are ties.
            
        self.action=self.reference.index[ind]
        self.Xnew=xfull.loc[self.action]

        return(self.action) #return the selected intervention name       
    
    def comp_std(self,model,xfull): #function for computing standard deviation
        preds = []
        for pred in model.estimators_:
            preds.append(pred.predict([xfull])[0])
        return(np.std(preds))
    
    def update(self,reward): #function for updating the RF model
        self.X.append(self.Xnew)
        self.Y.append(reward)
        self.RF.fit(self.X,self.Y)


# ### 1. importing data & preprocessing

# In[3]:


data=pd.read_csv('Poptherapy_dat.csv', header=0, sep=',', quotechar='"')


# In[4]:


data=data.drop(["Participant Key",'Allowed To Select Different Intervention',
    'InterventionIsSocial','InterventionIsPositivePsychology', 'InterventionIsCognitiveBehavioral',
    'InterventionIsEmotionalRegulation', 'InterventionIsSomatic',
 #'use ML 2 Select Intervention',
 'Suggested Intervention Time',
 #'Stress Before',
 'Stress After',
 'Selected Intervention Name',
 'day',
 'e','numberOfCalendarRecords','numberOfNotFreeCalendarRecords',
 'numberOfNotFreeCalendarRecordsBeforeIntervention',
 'numberOfNotFreeCalendarRecordsAtIntervention',
 'timeToNextNotFreeMeeting','numberOfGPSRecordsWithNullQuality',
 'qualityOfLastGPSRecord','timeSinceLastSelfReport','lunchTime',
 'nightTime','jerkX',
 'jerkY',
 'jerkZ','timeSinceLastAccelerationRecord','timePhoneUsed',
 'timePhoneUsedLast10Minutes','version',
 "intervention duration"],axis=1)
 


# In[5]:


len(data.columns)


# In[6]:


data=pd.get_dummies(data=data, columns=['dayOfWeek','employ','professionalLevel'], drop_first=False)


# remove highly correlated variables

# In[7]:


data=data.drop(['averageDistanceHome', 'minimalDistanceHome','averageDistanceWork', 'minimalDistanceWork','selfReportsEnergyAvg',
       'selfReportsMoodAvg','selfReportsMoodVariance','averageXaccelration', 'averageYaccelration',
       'averageZaccelration','averageYaccelration30minutes','averageZaccelration30minutes', 'varianceYaccelration30minutes',
       'varianceZaccelration30minutes', 'wasMarried', 'employStudent','professionalLevel_3.0','professionalLevel_99.0'],axis=1)


# In[8]:


len(data.columns)


# remove variables that cannot be collected in HS0 set up

# In[9]:


data=data.drop(['averageQualityOfGPSSignal','numberAccelerationRecords','averageXaccelration30minutes',
                  'numberAccelerationRecords30minutes','jerk30minutes','selfReportsEnergyVariance'],axis=1)


# In[10]:


len(data.columns)


# stratify "stress before"

# In[11]:


stressbefore=pd.qcut(data['Stress Before'],4,[1,2,3,4])

data=data.drop(['Stress Before'],axis=1)

data['Stress Before']=stressbefore


# stratify gps and mood

# In[12]:


distanceTraveled=pd.qcut(data['distanceTraveled'],2,[1,2])
lastDistanceHome=pd.qcut(data['lastDistanceHome'],2,[1,2])
lastDistanceWork=pd.qcut(data['lastDistanceWork'],2,[1,2])
lastReportedEnergy=pd.cut(data['lastReportedEnergy'],[-1.2,0.,data['lastReportedEnergy'].quantile(0.5),1.],labels=['missing',1,2])
lastReportedMood=pd.cut(data['lastReportedMood'],[-1.2,0.,data['lastReportedMood'].quantile(0.5),1.],labels=['missing',1,2])

data=data.drop(['distanceTraveled','lastDistanceHome','lastDistanceWork','lastReportedEnergy','lastReportedMood'],axis=1)

data['distanceTraveled']=distanceTraveled
data['lastDistanceHome']=lastDistanceHome
data['lastDistanceWork']=lastDistanceWork
data['lastReportedEnergy']=lastReportedEnergy
data['lastReportedMood']=lastReportedMood


# In[13]:


data=pd.get_dummies(data=data, columns=['Stress Before', 'distanceTraveled','lastDistanceHome','lastDistanceWork','lastReportedEnergy','lastReportedMood'], drop_first=False)


# In[14]:


len(data.columns)


# In[15]:


data.columns


# ### Offline evaluation

# In[16]:


#use only random data when doing offline evaluation (Li et al., 2011)

data_random=data[(data["use ML 2 Select Intervention"]==False)] 

#make reference table
reference=pd.DataFrame(np.zeros((8,5)),columns=['InterventionIsSocial','InterventionIsPositivePsychology', 'InterventionIsCognitiveBehavioral',
                                                               'InterventionIsEmotionalRegulation', 'InterventionIsSomatic'],
                                                    index=['Soul food - Me','Mindfulness - Me','Happy hearth - Me','Body health - Me',
                                                        'Soul food - Social','Mindfulness - Social','Happy hearth - Social','Body health - Social'])
reference['InterventionIsSocial']['Soul food - Social','Mindfulness - Social','Happy hearth - Social','Body health - Social']=1.
reference['InterventionIsPositivePsychology']['Soul food - Me','Soul food - Social']=1.
reference['InterventionIsCognitiveBehavioral']['Mindfulness - Me','Mindfulness - Social']=1.
reference['InterventionIsEmotionalRegulation']['Happy hearth - Me','Happy hearth - Social']=1.
reference['InterventionIsSomatic']['Body health - Me','Body health - Social']=1.


# In[17]:


reference


# In[18]:


np.random.seed(0)

n_simul=1
T=400

cumulated_reward_UCB_1=[]
cumulated_reward_UCB_2=[]
cumulated_reward_UCB_3=[]
cumulated_reward_random=[]

for simul in range(n_simul):

    M1=RF_UCB(v=0.1,reference=reference); T1=0
    M2=RF_UCB(v=0.5,reference=reference); T2=0
    M3=RF_UCB(v=1.,reference=reference); T3=0
    RWD1=[]
    RWD2=[]
    RWD3=[]
    RWD_random=[]; Tr=0
    notdone=True
    
    while notdone:
        
        idx=np.random.choice(range(len(data_random)))
        rwd=data_random.iloc[idx]["stress delta"]
        x=data_random.iloc[idx].drop(["stress delta",'use ML 2 Select Intervention',"Suggested Intervention Name"])
        treatment=data_random.iloc[idx]["Suggested Intervention Name"]
        
        if T1<T:        
            a1=M1.select_ac(x,T1)
            if treatment==a1:
                RWD1.append(rwd)
                M1.update(rwd)
                T1+=1
                print(T1)
        
        if T2<T:
            a2=M2.select_ac(x,T2)
            if treatment==a2:
                RWD2.append(rwd)
                M2.update(rwd)
                T2+=1
            
        if T3<T:
            a3=M3.select_ac(x,T3)
            if treatment==a3:
                RWD3.append(rwd)
                M3.update(rwd)
                T3+=1
        
        if Tr<T:
            a4=np.random.choice(reference.index)
            if treatment==a4:
                RWD_random.append(rwd)
                Tr+=1
            
        if T1==T:
            if T2==T:
                if T3==T:
                    if Tr==T:
                        break
                        
    cumulated_reward_UCB_1.append(np.cumsum(RWD1))
    cumulated_reward_UCB_2.append(np.cumsum(RWD2))
    cumulated_reward_UCB_3.append(np.cumsum(RWD3))
    cumulated_reward_random.append(np.cumsum(RWD_random))
    


# In[20]:


steps=np.arange(1,T+1)
plt.plot(steps,np.median(cumulated_reward_UCB_1,axis=0)[:T],'b',label='UCB_1')
plt.plot(steps,np.percentile(cumulated_reward_UCB_1,25,axis=0)[:T],'b',linestyle='--')
plt.plot(steps,np.percentile(cumulated_reward_UCB_1,75,axis=0)[:T],'b',linestyle='--')
plt.plot(steps,np.median(cumulated_reward_UCB_2,axis=0)[:T],'r',label='UCB_2')
plt.plot(steps,np.percentile(cumulated_reward_UCB_2,25,axis=0)[:T],'r',linestyle='--')
plt.plot(steps,np.percentile(cumulated_reward_UCB_2,75,axis=0)[:T],'r',linestyle='--')
plt.plot(steps,np.median(cumulated_reward_UCB_3,axis=0)[:T],'g',label='UCB_3')
plt.plot(steps,np.percentile(cumulated_reward_UCB_3,25,axis=0)[:T],'g',linestyle='--')
plt.plot(steps,np.percentile(cumulated_reward_UCB_3,75,axis=0)[:T],'g',linestyle='--')
plt.plot(steps,np.median(cumulated_reward_random,axis=0)[:T],'m',label='random')
plt.plot(steps,np.percentile(cumulated_reward_random,25,axis=0)[:T],'m',linestyle='--')
plt.plot(steps,np.percentile(cumulated_reward_random,75,axis=0)[:T],'m',linestyle='--')


plt.xlabel('Decision Point')
plt.ylabel('Cumulative Reward')
plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.2),fancybox=True,ncol=5)
plt.show()

