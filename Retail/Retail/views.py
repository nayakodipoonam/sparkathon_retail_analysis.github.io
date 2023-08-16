from django.http import HttpResponse
from django.shortcuts import render

#Loading neccesary packages
import os
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


def home(request):
    return HttpResponse("hello")
    
def recommend(request):
    #Reading Data From Web
    if request.method == 'POST':
        item = request.POST.get('product')
        my_data = pd.read_csv('E:/E-old/sparkathon/dataset/archive/bread_basket_updated.csv')
        # my_data.head(10)
        # print(my_data)
        
        # Data Cleaning
        my_data['Item'] = my_data['Item'].str.strip() #removes spaces from beginning and end
        my_data.dropna(axis=0, subset=['Transaction'], inplace=True) #removes duplicate invoice
        my_data['Transaction'] = my_data['Transaction'].astype('str') #converting invoice number to be string
        my_data = my_data[~my_data['Transaction'].str.contains('C')] #remove the credit transactions

        # my_data.head()

        # my_data['Country'].value_counts()
        # print(my_data)

        #Separating transactions for United Kingdom
        mybasket = (my_data
            .groupby(['Transaction', 'Item'])['Quantity']
            .sum().unstack().reset_index().fillna(0)
            .set_index('Transaction'))


        # print(mybasket)


        #converting all positive values to 1 and everything else to 0
        def my_encode_units(x):
            if x <= 0:
                return 0
            if x >= 1:
                return 1

        my_basket_sets = mybasket.applymap(my_encode_units)
        # my_basket_sets.drop('POSTAGE', inplace=True, axis=1) #Remove "postage" as an item


        # Model Training

        #Generatig frequent itemsets
        my_frequent_itemsets = apriori(my_basket_sets, min_support=0.03, use_colnames=True)
        # print(my_frequent_itemsets)

        #generating rules
        my_rules = association_rules(my_frequent_itemsets, metric="lift", min_threshold=0.5)

        # print(my_rules)


        # Making Recommendations

        #Filtering rules based on condition
        my_rules[ (my_rules['lift'] >= 1) &
            (my_rules['confidence'] >= 0.3) ]


        
        filtered = my_rules[my_rules["antecedents"] == frozenset({item})]["consequents"]
        items = []

        for data in filtered:
            items.extend(data)
        return render(request,"result.html",{"list":items})
    
    return render(request,"index.html")


def sample(request):
    return render(request,"sample.html")

        

