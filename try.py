#Loading neccesary packages
import os
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#Reading Data From Web
# myretaildata = pd.read_csv('E:\E-old\sparkathon\dataset\grocery\Supermart Grocery Sales - Retail Analytics Dataset.csv', engine='openpyxl')
myretaildata = pd.read_csv('E:/E-old/sparkathon/dataset/archive/bread_basket_updated.csv')
# myretaildata.head(10)
# print(myretaildata)


#Data Cleaning
myretaildata['Item'] = myretaildata['Item'].str.strip() #removes spaces from beginning and end
myretaildata.dropna(axis=0, subset=['Transaction'], inplace=True) #removes duplicate invoice
myretaildata['Transaction'] = myretaildata['Transaction'].astype('str') #converting invoice number to be string
myretaildata = myretaildata[~myretaildata['Transaction'].str.contains('C')] #remove the credit transactions



# print(myretaildata)

#Separating transactions for United Kingdom
mybasket = (myretaildata
          .groupby(['Transaction', 'Item'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('Transaction'))

# print(mybasket)

# for i in mybasket:
#     if i

# print(mybasket.size)


#converting all positive values to 1 and everything else to 0
def my_encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

my_basket_sets = mybasket.applymap(my_encode_units)
# my_basket_sets.drop('POSTAGE', inplace=True, axis=1) #Remove "postage" as an item

# print(my_basket_sets)
# Model Training

#Generatig frequent itemsets
my_frequent_itemsets = apriori(my_basket_sets, min_support=0.03, use_colnames=True)
print(len(my_frequent_itemsets))
print(my_frequent_itemsets)


#generating rules
my_rules = association_rules(my_frequent_itemsets, metric="lift", min_threshold=0.5)
print(len(my_rules))
print(my_rules)




# Making Recommendations

#Filtering rules based on condition
my_rules[ (my_rules['lift'] >= 3) &
       (my_rules['confidence'] >= 0.3) ] 


item="ROUND SNACK BOXES SET OF 4 FRUITS"
filtered = my_rules[my_rules["antecedents"] == frozenset({'Coffee'})]["consequents"]
# filtered.size
# print(my_rules.size)
# print(my_rules)
# print(my_rules['antecedents']=='Bread')


print("here are your recommended items for  "+item+"--")
print(filtered)


items = []

for data in filtered:
    items.extend(data)

# print(items[0])
