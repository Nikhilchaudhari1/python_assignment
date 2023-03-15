#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Write a Python program which accepts a sequence of comma-separated numbers from user and generate a list and a tuple with those numbers.

values = input("Input some comma seprated numbers : ")
list = values.split(",")
tuple = tuple(list)
print('List : ',list)
print('Tuple : ',tuple)


# In[2]:


# Write a Python program to display the first and last colors from the following list.

color_list = ["Red","Green","White" ,"Black"]
print( "%s %s"%(color_list[0],color_list[-1]))


# In[3]:


# Write a Python program to print the even numbers from a given list.

list1 = [1,2,3,4,5,6,7,8,9]
for num in list1: 
    if num % 2 == 0:
        print(num, end=" ")


# In[4]:


# Write a Python program to calculate number of days between two dates. Hint: use Datetime package/module.

from datetime import date
f_date = date(2014, 7, 2)
l_date = date(2014, 7, 11)
delta = l_date - f_date
print(delta.days)


# In[5]:


# Write a Python program to get the volume of a sphere with radius 6.

pi = 3.1415926535897931
r= 6
V= 4/3*pi* r**3
print('The volume of the sphere is: ',V)


# In[7]:


# Write a Python program to count the number 4 in a given list.

def list_count_4(nums):
  count = 0  
  for num in nums:
    if num == 4:
      count = count + 1
  return count
print(list_count_4([1, 4, 6, 8, 4, 9, 4]))


# In[10]:


# Write a Python program to print all even numbers from a given numbers list in the same order and stop the printing if any numbers that come after 237 in the sequence. Go to the editorSample numbers list :

numbers = [399, 162, 758, 219, 918, 237, 412, 566, 826, 248, 866, 950, 626, 949, 687, 217,

           815, 67, 104, 58, 512, 24, 892, 894, 767, 553, 81, 379, 843, 831, 445, 742, 717,

           958,743, 527]
    
def Even_number(lists):
    for a in lists:
        if a % 2 == 0:
            print(a)
        elif a == 237:
            break
            
Even_number(numbers)


# In[11]:


# Write a Python program to find those numbers which are divisible by 7 and multiple of 5, between 1500 and 2700 (both included)

nl=[]
for x in range(1500, 2700):
    if (x%7==0) and (x%5==0):
        nl.append(str(x))
print (','.join(nl))


# In[12]:


# Write a Python program that prints all the numbers from 0 to 6 except 3 and 6.

for x in range(6):
    if (x == 3 or x==6):
        continue
    print(x,end=' ')
print("\n")


# In[13]:


# Write a Python program to get the Fibonacci series between 0 to 50.

x,y=0,1
while y<50:
    print(y)
    x,y = y,x+y


# In[14]:


#Write a Python function that takes a list and returns a new list with unique elements of the first list.
def unique_list(l):
  x = []
  for a in l:
    if a not in x:
      x.append(a)
  return x
print(unique_list([1,2,3,3,3,3,4,5])) 


# In[15]:


# Write a Python program to concatenate all elements in a list into a string and return it.
def concatenate_list_data(list):
    result= ''
    for element in list:
        result += str(element)
    return result
 
print(concatenate_list_data([5, 27, 11, 3]))


# In[16]:


# Write a Python script to concatenate following dictionaries to create a new one.

dic1={1:10, 2:20}  
dic2={3:30, 4:40}  
dic3={5:50,6:60}  
dic4 = {}  
for d in (dic1, dic2, dic3): dic4.update(d)  
print(dic4) 


# In[17]:


# Write a Python program to add, subtract, multiple and divide two Pandas Series.

import pandas as pd
a = pd.Series([2, 4, 6, 8, 10])
b = pd.Series([1, 3, 5, 7, 9])
c = a + b
print("Add two Series:")
print(c)
print("Subtract two Series:")
c = a - b
print(c)
print("Multiply two Series:")
c = a * b
print(c)
print("Divide Series1 by Series2:")
c = a / b
print(c)


# In[18]:


'''Write a Pandas program to select the specified columns and rows from a given data frame. Go to the editorSample Python dictionary data and list labels:

Select 'name' and 'score' columns in rows 1, 3, 5, 6 from the following data frame.

exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],

score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],

attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

Expected Output:

Select specific columns and rows:

name score

b Dima 9.0

d James NaN

f Michael 20.0

g Matthew 14.5 '''
            
import numpy as np
import pandas as pd
exam_data = {'name': ['Anastasia','Dima','Katherine','James','Emily','Michael','Mathew','Laura','Kevin','Jonas'],
           'score': [12.5,9,16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
           'attempts': [1,3,2,3,2,3,1,1,2,1],
           'qualify': ['yes','no','yes','no','no','yes','yes','no','no','yes']}
labels = ['a','b','c','d','e','f','g','h','i','j']

data = pd.DataFrame(exam_data, index=labels)
data
      
            


# In[19]:


print("Select specific columns and rows:")
print(data.iloc[[1, 3, 5, 6], [0, 1]])


# In[23]:


''' Use Crime dataset from LMS

I) find the aggregations like all moments of business decisions for all columns,value counts.

II) do the plottings like plottings like histogram, boxplot, scatterplot, barplot, piechart,dot chart.'''

import pandas as pd
crime_dataset = pd.read_csv("C:/Users/nikhil chaudhari/Desktop/crime_data.csv")
crime_dataset.rename({'Unnamed: 0': 'City'}, axis = 1, inplace=True)
crime_dataset.describe()


# In[24]:


import seaborn as sns
import matplotlib.pyplot as plt
features = crime_dataset.columns[1:].tolist()
for feat in features:
    skew = crime_dataset[feat].skew()
    sns.distplot(crime_dataset[feat], kde= False, label='Skew = %.3f' %(skew), bins=10)
    plt.legend(loc='best')
    plt.show()


# In[25]:


ot=crime_dataset.copy() 
fig, axes=plt.subplots(4,1,figsize=(12,8),sharex=False,sharey=False)
sns.boxplot(x='Murder',data=ot,palette='crest',ax=axes[0])
sns.boxplot(x='Assault',data=ot,palette='crest',ax=axes[1])
sns.boxplot(x='UrbanPop',data=ot,palette='crest',ax=axes[2])
sns.boxplot(x='Rape',data=ot,palette='crest',ax=axes[3])
plt.tight_layout(pad=2.0)


# In[26]:


sns.pairplot(crime_dataset)


# In[27]:


crime_dataset.columns


# In[28]:


plt.figure(figsize=(20,8))
# make barplot and sort bars
sns.barplot(x='City',
            y="Murder", 
            data=crime_dataset, 
            order=crime_dataset.sort_values('Murder').City)
# set labels
plt.xlabel("City", size=15)
plt.ylabel("Murder Rate", size=15)
plt.yticks(rotation = 0, fontsize = 14)
plt.xticks(rotation = 90, fontsize = 16)
plt.title("Murder Rate in US City wise", size=18)
plt.show()

plt.figure(figsize=(20,8))
# make barplot and sort bars
sns.barplot(x='City',
            y="UrbanPop", 
            data=crime_dataset, 
            order=crime_dataset.sort_values('UrbanPop').City)
# set labels
plt.xlabel("City", size=15)
plt.ylabel("Urban Population Rate", size=15)
plt.yticks(rotation = 0, fontsize = 14)
plt.xticks(rotation = 90, fontsize = 16)
plt.title("Urban Population Rate in US City wise", size=18)
plt.show()

plt.figure(figsize=(20,8))
# make barplot and sort bars
sns.barplot(x='City',
            y="Assault", 
            data=crime_dataset, 
            order=crime_dataset.sort_values('Assault').City)
# set labels
plt.xlabel("City", size=15)
plt.ylabel("Assault Rate", size=15)
plt.yticks(rotation = 0, fontsize = 14)
plt.xticks(rotation = 90, fontsize = 16)
plt.title("Assault Rate in US City wise", size=18)
plt.show()

plt.figure(figsize=(20,8))
# make barplot and sort bars
sns.barplot(x='City',
            y="Rape", 
            data=crime_dataset, 
            order=crime_dataset.sort_values('Rape').City)
# set labels
plt.xlabel("City", size=15)
plt.ylabel("Rape Rate", size=15)
plt.yticks(rotation = 0, fontsize = 14)
plt.xticks(rotation = 90, fontsize = 16)
plt.title("Rape Rate in US City wise", size=18)
plt.show()


# In[29]:


murder = crime_dataset.sort_values('Murder', ascending = False, ignore_index=True)

plt.figure(figsize = (8,8))
plt.pie(murder.Murder[:10],
       labels=murder.City[:10],
       explode = [0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
       autopct= '%.2f%%',
       shadow= True,
       startangle= 190,
       textprops = {'size':'large',
                   'fontweight':'bold',
                    'rotation':'0',
                   'color':'black'})
#plt.legend(loc= 'best')
plt.title("Top 10 Highest Murder Rate City Wise", fontsize = 18, fontweight = 'bold')
plt.show()

murder = crime_dataset.sort_values('Murder', ascending = True, ignore_index=True)

plt.figure(figsize = (8,8))
plt.pie(murder.Murder[:10],
       labels=murder.City[:10],
       explode = [0.2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
       autopct= '%.2f%%',
       shadow= True,
       startangle= 190,
       textprops = {'size':'large',
                   'fontweight':'bold',
                    'rotation':'0',
                   'color':'black'})
#plt.legend(loc= 'best')
plt.title("Top 10 Lowest Murder Rate City-Wise", fontsize = 18, fontweight = 'bold')
plt.show()

assault = crime_dataset.sort_values('Assault', ascending = False)

plt.figure(figsize = (8,8))
plt.pie(assault.Assault[:10],
       labels=assault.City[:10],
       explode = [0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
       autopct= '%.2f%%',
       shadow= True,
       startangle= 190,
       textprops = {'size':'large',
                   'fontweight':'bold',
                    'rotation':'0',
                   'color':'black'})
#plt.legend(loc= 'best')
plt.title("Top 10 Highest Assault Rate City-wise", fontsize = 18, fontweight = 'bold')
plt.show()

assault = crime_dataset.sort_values('Assault', ascending = True)

plt.figure(figsize = (8,8))
plt.pie(assault.Assault[:10],
       labels=assault.City[:10],
       explode = [0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
       autopct= '%.2f%%',
       shadow= True,
       startangle= 190,
       textprops = {'size':'large',
                   'fontweight':'bold',
                    'rotation':'0',
                   'color':'black'})
#plt.legend(loc= 'best')
plt.title("Top 10 Lowest Assault Rate City-wise", fontsize = 18, fontweight = 'bold')
plt.show()

Rape = crime_dataset.sort_values('Rape', ascending = False)

plt.figure(figsize = (8,8))
plt.pie(Rape.Rape[:10],
       labels=Rape.City[:10],
       explode = [0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
       autopct= '%.2f%%',
       shadow= True,
       startangle= 190,
       textprops = {'size':'large',
                   'fontweight':'bold',
                    'rotation':'0',
                   'color':'black'})
#plt.legend(loc= 'best')
plt.title("Top 10 Highest Rape Rate City-wise", fontsize = 18, fontweight = 'bold')
plt.show()

Rape = crime_dataset.sort_values('Rape', ascending = True)

plt.figure(figsize = (8,8))
plt.pie(Rape.Rape[:10],
       labels=Rape.City[:10],
       explode = [0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
       autopct= '%.2f%%',
       shadow= True,
       startangle= 190,
       textprops = {'size':'large',
                   'fontweight':'bold',
                    'rotation':'0',
                   'color':'black'})
#plt.legend(loc= 'best')
plt.title("Top 10 Lowest Rape Rate City-wise", fontsize = 18, fontweight = 'bold')
plt.show()


# In[40]:


import pandas as pd
df = pd.read_csv("C:/Users/nikhil chaudhari/Desktop/mtcars.csv")


# In[ ]:


'''3. use mtcars dataset from LMS

A) delete/ drop rows-10 to 15 of all columns

B)drop the VOL column

C)write the forloop to get value_counts of all cloumns'''


# In[ ]:


df.drop(index = list(range(10,16)))


# In[ ]:


df.drop(columns = 'VOL')


# In[47]:


for column in df.columns:
    print("Column Name: ", column)
    print(df[column].value_counts())


# In[ ]:


'''4. Use Bank Dataset from LMS

A)change all the categorical columns into numerical by creating Dummies and using label encoder.

B) rename all the column names DF

C) Rename only one specific column in DF'''


# In[53]:


#A).
import pandas as pd
dataframe = pd.read_csv("C:/Users/nikhil chaudhari/Desktop/bank-full.csv", delimiter=';')


# In[54]:


df = dataframe.copy()
# One-Hot Encoding of categrical variables
df=pd.get_dummies(df,columns=['job', 'marital', 'education', 'contact', 'month', 'poutcome'])
# To see all columns
pd.set_option("display.max.columns", None)
df


# In[55]:


#B.

print('Old column names: ',dataframe.columns)
print('_'*100+'\n'+'_'*100+'\n')
dataframe.rename({'age':'Age','job':'Jobu',                  'marital':'Marital',                  'education':'Education',                  'default':'Default',                  'balance':'Balance',                  'housing':'Housing',                  'loan':'Loan',                  'contact':'Contact',                  'day':'Day',                  'month':'Month',                  'duration':'Duration',                  'campaign':'Campaign',                  'pdays':'Pdays',                  'previous':'Previous',                  'poutcome':'POutcome',                  'y':'Subscription'}, axis =1 , inplace = True)
print('New column names: ',dataframe.columns)


# In[56]:


#C.

dataframe.rename({'Age':'age'}, axis =1 , inplace = True)
dataframe.columns


# In[57]:


# After doing all the changes in bank data(Q19). save the file in your directory in Csv Format.


df.to_csv('crime_dataset_preprocessed.csv', index=False)


# In[59]:


#1. Write Python Programs to use various operators in Python.

def various_operators(list_1, list_2, function):
    list_3 = []
    if len(list_1) != len(list_2):
        print("Please provide both lists with equal length")
    elif function == 'add':
        for i in range(len(list_1)):
            list_3.append(list_1[i] + list_2[i])
        return list_3
    elif function == 'subtract':
        for i in range(len(list_1)):
            list_3.append(list_1[i] - list_2[i])
        return list_3
    elif function == 'multiply':
        for i in range(len(list_1)):
            list_3.append(list_1[i] * list_2[i])
        return list_3
    elif function == 'divide':
        for i in range(len(list_1)):
            list_3.append(round(list_1[i] / list_2[i],2))
        return list_3
    elif function == 'modulo':
        for i in range(len(list_1)):
            list_3.append(round(list_1[i] % list_2[i],2))
        return list_3
    elif function == 'square':
        list_4 = []
        for i in range(len(list_1)):
            list_3.append(round((list_1[i])**2,2))
            list_4.append(round((list_2[i])**2,2))
        return list_3 + list_4
    elif function == 'cube':
        list_4 = []
        for i in range(len(list_1)):
            list_3.append(round((list_1[i])**3,2))
            list_4.append(round((list_2[i])**3,2))
        return list_3 + list_4
    else:
        print('Please provide a valid arithmetic operator like "add",               "subtract", "multiply", divide", "module", "square" and "cube"')


# In[60]:


#2. Create list of elements and slice and dice it

# Initialize list
Lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
 
# Display list
print(Lst[::2])

# Display list
print(Lst[1::2])

# Display list
print(Lst[:5])

# Display list
print(Lst[-5:])

# Display list
print(Lst[6:])


# In[61]:


#3. Using while loop accept numbers until sum of numbers is less than 100.

mySum = 0
num = 0
while mySum < 100:
    num += 1
    theSum = num
    mySum = theSum + num

print(num, theSum, mySum)


# In[ ]:


#4. Write a python program Read & write Excel files.

from openpyxl import Workbook
import time

book = Workbook()
sheet = book.active

sheet['A1'] = 'Name'
sheet['A2'] = 'Dalvi Moin'

now = time.strftime("%x")
sheet['A3'] = now

book.save("sample.xlsx")


# In[62]:


#6. Create a 3x3 matrix with values ranging from 2 to 10 using numpy.

import numpy as np
x =  np.arange(2, 11).reshape(3,3)
print(x)


# In[63]:


#7. Write a Python program to convert a list of numeric value into a one-dimensional NumPy array.

import numpy as np
print("Original List : ",[12.23, 13.32, 100, 36.32])
a = np.array([12.23, 13.32, 100, 36.32])
print("One-dimensional NumPy array : ",a)


# In[64]:


#8. "Write a Python program to create a null vector of size 10 and update sixth value to 11.

import numpy as np
x = np.zeros(10)
print(x)
print("Update sixth value to 11")
x[6] = 11
print(x)


# In[ ]:




