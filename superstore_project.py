#Бизнес-вопросы:
print('''1.Какие категории дают максимальную выручку?
2.Какие регионы самые прибыльные?
3.Какие товары имеют высокий спрос, но низкую прибыль?
4.Как меняются продажи во времени?
5.Кто основные клиенты?''')



#крч библеотеки импортируем
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#настройки изображения
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', '{:.2f}'.format)



#1.DATA SET DOWNLOAD
df = pd.read_csv(r'C:\Users\User\Downloads\train_a.csv.zip')
print("\n" + "="*50)
print('ЗАГРУЗКА ДАННЫХ')
print("="*50)
print(df.head())
print('\nИнформация о датафрейме:')
print(df.info())
print('\nТипы данных:')
print(df.dtypes)
print('\nСтатистика по числовым колонкам:')
print(df.describe())

#2. DATA CLEANING
print("\n" + "="*50)
print('ОЧИСТКА ПРОПУСКОВ И ДУБЛКАТОВ')
print("="*50)
#2.1 работа со строками
df.columns = df.columns.str.lower().str.replace({' ': '_', '-':'_'})
df['ship_mode'] = df['ship_mode'].str.replace({' ': '_', '-':'_'})
text = df.select_dtypes(include = ['object','string']).columns
text_1 = [col for col in text if col != 'product_name']
for col in text_1:
    df[col] = df[col].str.strip().str.lower()

print(df.head())
#2.2 копии и пропуски
df = df.drop(columns = ['country'])
print('\nКоличество дубликатов:')
print(df.duplicated().sum())
print('\nКоличество пропусков:')
print(df.isna().sum())
#заполнение одного пропуска
df['postal_code'] = df['postal_code'].fillna(df.groupby('city')['postal_code'].transform(lambda x: x.mode()[0]))
print('\nУникальных значений:')
print(df.nunique())
#2.3 преобразование данных
df['category'] = df['category'].astype('category')
df['sub_category'] = df['sub_category'].astype('category')


df['order_date'] = pd.to_datetime(df['order_date'],dayfirst = True)
df['ship_date'] = pd.to_datetime(df['ship_date'],dayfirst = True)
df['month'] = df['order_date'].dt.month
df['year'] = df['order_date'].dt.year

print(df.info())
print(df.head(100))
print(df.nunique())
print(df['city'].value_counts())
#2.4 выбросы

def find_outler(col):
    q1 = col.quantile(0.25)
    q3 = col.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outler = col[
        (col < lower) |
        (col > upper)]
    return outler
print(find_outler(df['sales']))


#3 EDA
print("\n" + "="*50)
print('AНАЛИЗИРОВАНИЕ ПРИЗНАКОВ')
print("="*50)

#3.1 одномерный анализ
print("\n" + "="*50)
print('AНАЛИЗИРОВАНИЕ ПРИЗНАКОВ ОДНОМЕРНЫХ')
print("="*50)

#АНАЛИЗ ЧИСЛОВЫХ ПРИЗНАКОВ 
print(df['sales'].mean())
print(df['sales'].median())
print(df.sort_values('sales',ascending = False))


#ОБЩЕЕ РАСПРЕДЕЛЕНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ

#анализ категорий
print('\nРаспределение категорий:',df['category'].value_counts())
procent_of_category_count = df['category'].value_counts() / len(df) * 100
print('\nКоличественное распределение категорий по процентам:',procent_of_category_count)
#анализ подкатегорий
print('\nРаспределение категорий по подкатегориям',df['sub_category'].value_counts())
procent_of_subcategory_count = df['sub_category'].value_counts() / len(df) * 100
print('\nКоличественное распределение подкатегорий по процентам:',procent_of_subcategory_count)

#анализ городов и регионов
#1
print('\nРаспределение штатов:',df['state'].value_counts())
procent_of_state_count = df['state'].value_counts() / len(df) * 100
print('\nКоличественное распределение штатов по процентам:',procent_of_state_count)
#2
print('\nРаспределение регионов:',df['region'].value_counts())
procent_of_region_count = df['region'].value_counts() / len(df) * 100
print('\nКоличественное распределение регионов по процентам:',procent_of_region_count)
#3
print('\nРаспределение городов:',df['city'].value_counts())
procent_of_city_count = df['city'].value_counts() / len(df) * 100
print('\nКоличественное распределение городов по процентам:',procent_of_city_count)

#анализ сегметов
#1
print('\nРаспределение cегментов :',df['segment'].value_counts())
procent_of_segment_count = df['segment'].value_counts() / len(df) * 100
print('\nКоличественное распределение сегментов по процентам:',procent_of_segment_count)
#2
print('\nРаспределение доставки:',df['ship_mode'].value_counts())
procent_of_ship_mode_count = df['ship_mode'].value_counts() / len(df) * 100
print('\nКоличественное распределение доставки по процентам:',procent_of_ship_mode_count)


#АНАЛИЗ C ПОМОЩЬЮ ВИЗУАЛИЗАЦИИ
fig, axes = plt.subplots(3,3,figsize = (12,16))
sns.histplot(data = df, x = 'sales',bins = 50,ax = axes[0,0])
axes[0,0].set_title('распределение цен')

sns.barplot(data = df, x = 'region',y = 'sales',  ax = axes[0,1])
axes[0,1].set_title('распределение регионов')

sns.ecdfplot(data = df, x = 'sales', hue = 'region', ax = axes[1,0])

sns.barplot(data = df, x = 'category', y = 'sales', ax =axes[1,1])
sns.lineplot(data = df,x = 'year',y = 'sales', ax = axes[0,2])
plt.tight_layout()
plt.show()

                                     
procent_of_category_sales = df.groupby('category')['sales'].sum() / df['sales'].sum() * 100
procent_of_subcategory_sales = df.groupby('sub_category')['sales'].sum() / df['sales'].sum() * 100












