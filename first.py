import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


diary = pd.DataFrame({
    'day':['mon','tue','wed','thu','fri','sat','sun'],
    'sleep_hour':[6,3,8,6,8,9,7],
    'study_hour':[4,2,3,2,4,5,4],
    'sport_minutes': [90,20,90,40,80,30,30],
    'mood':[6,3,5,2,7,8,9],
    'phone_hours': [ 8,2,5,6,8,9,8]
    })

diary['score'] = diary.apply(lambda row:
                     row['study_hour'] * 2 + row['sport_minutes'] / 60 - row['phone_hours'], axis = 1)
print('your balance: \n',diary.sort_values('score', ascending = False))
#баланс

def mood_type(row):
    if row['mood'] >= 8:
        return 'good day'
    elif row['mood'] >= 5:
        return 'normal'
    else:
        return 'bad day'
    
diary['day_type'] = diary.apply(mood_type,axis = 1)
print('bad days is: \n',diary[diary['day_type'] == 'bad day'][['day','day_type']])
#плохие дни и хорошие

def sleep_status(row):
    if row['sleep_hour'] < 6:
        return 'tired'
    else:
        return 'rested'
diary['activ'] = diary.apply(sleep_status,axis = 1)
print('tired day is: \n',diary[diary['activ'] == 'tired'][['day','activ']])
#дни активности и усталости

def pro(row):
    if row['study_hour'] > 3 and row['mood'] > 6:
        return 'productiv boy'
    else :
        return 'nice'
    
diary['productiv'] = diary.apply(pro,axis = 1)
print('productiv days are:\n ',diary[diary['productiv'] == 'productiv boy'][['day','productiv']])
#продуктивные дни

print('good mood is: \n',diary.groupby('sleep_hour').filter(lambda x: x['mood'].mean() > 6))
#тут мы считаем в какте дни настроение лучще

print('high level of phone watching: \n',diary.groupby('study_hour').filter(
    lambda x: x['phone_hours'].mean() > 4).sort_values('phone_hours',ascending = False))
 #считаем в какие дни тедефонная норма сильна превыщена









    
