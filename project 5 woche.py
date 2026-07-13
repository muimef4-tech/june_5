import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', '{:.2f}'.format)
print(os.getcwd())
df = pd.read_csv(r'C:\Users\User\Downloads\netflix_titles.csv.zip' )
#очистка данных и преобразование
print(df.head())
print(df.info())
print(df.dtypes)
print(df.describe())
print(df[df.duplicated])
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.strip().str.lower()
df['date_added'] = pd.to_datetime(df['date_added'], dayfirst=True, errors ='coerce')  
df['director'] = df['director'].fillna('неизвестно')
df['cast'] = df['cast'].fillna('неизвестно')
df['country'] = df['country'].fillna('неизвестно')
df = df.dropna(subset = ['date_added'])
df['rating'] = df['rating'].fillna('no rating')
print(df.isna().sum())
print(df.nunique())
a = df[df['date_added'].isna()]
print(df.dtypes)
print(a)
df['country'] = df['country'].str.split(',').str[0]
#выбросы



#считаем EDA

popular_type = df['type'].value_counts()
#популярность типов фильма
top_10_COUNNTRY = df.groupby(['country'])['show_id'].count().sort_values(ascending = False).head(10)
#топ 10 стран по выпуску контента
t10_procent = (top_10_COUNNTRY / len(df['country']) * 100)
#доля топ 10 стран в общем выпуске контента
df['year'] = df['date_added'].dt.year
df['month'] = df['date_added'].dt.month
df['week'] = df['date_added'].dt.dayofweek
change_by_year = df.groupby('year')['show_id'].count().reset_index()
df = df.sort_values('month')
change_by_month = df.groupby('month')['show_id'].count().reset_index()
change_by_week = df.groupby('week')['show_id'].count().reset_index()
#количество контента по времени
rating_oft = df.groupby('rating')['show_id'].count().sort_values(ascending = False).head(1)
t1_procent = rating_oft / len(df) * 100
t1_category_age_procent = round(t1_procent.sum().sum())
top = df['rating'].value_counts().idxmax()
print(f'из {len(df)} ,{rating_oft.sum()} смотрят фильмы категории {top}')
#оценка рейтинга , а также количество контента по возрастному рейтингу

#сегментация
def content_age(year):
    if year >= 2020:
        return 'new'
    elif year >= 2010:
        return 'medium'
    else:
        return 'old'
    
df['content_ages'] =df['release_year'].apply(content_age)
print(df['content_ages'].value_counts())


movies = df[df['type'] == 'Movie'].copy()
movies['duration_time_m'] = (
    movies['duration'].str.extract('(\d+)').astype(float)
    )

print(movies['duration_time_m'])
def length_segment(minutes):
    if minutes < 90:
        return 'short'
    elif minutes <= 120:
        return 'standart'
    else:
        return 'long'

movies['length_segment_movies'] = movies['duration_time_m'].apply(length_segment)
print(movies['length_segment_movies'].value_counts())

shows = df[df['type'] == 'TV Show'].copy()
shows['duration_time_s'] = (
    shows['duration'].str.extract('(\d+)').astype(float)
    )
print(shows['duration_time_s'])
def length_segment(series):
    if series == 1:
        return 'short'
    elif series < 3:
        return 'standart'
    elif series < 5:
        return 'long'
    else:
        return 'very_long'

shows['shows_length_segment'] = shows['duration_time_s'].apply(length_segment)
print(shows['shows_length_segment'].value_counts())







def age_segment(rating):
    if rating in ['G', 'TV-Y', 'TV-Y7']:
        return 'Kids'
    elif rating in ['PG', 'PG-13', 'TV-G', 'TV-PG', 'TV-14']:
        return 'Teen'
    elif rating in ['R', 'TV-MA', 'NC-17']:
        return 'Adult'
    else:
        return 'Unknown'

df['age_category'] = df['rating'].apply(age_segment)
print(df['age_category'].value_counts())


df['show'] = df.groupby('country')['country'].transform('count')
df['segment_content'] = pd.cut(
    df['show'],
    bins = [0,100,500,float('inf')],
    labels = ['low','mid','high']
    )
print(df[['country','segment_content']])
print(df.groupby('segment_content')['show_id'].count())
#этап визуализации


print(movies.describe())
q1 = movies['duration_time_m'].quantile(0.25)
q3 =  movies['duration_time_m'].quantile( 0.75)
iqr = q3 - q1
a = movies[
    (movies['duration_time_m'] < q1 - 1.5 * iqr) |
    (movies['duration_time_m'] > q3 + 1.5 * iqr)]

print(a.groupby('aa')['ajk'].sum())





















