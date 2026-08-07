#Бизнес-вопросы:
print('''
1. Какие категории приносят максимальную выручку?

2. Какие подкатегории являются лидерами по продажам?

3. Какие товары имеют самый высокий спрос?

4. Какие регионы приносят больше всего выручки?

5. Какие штаты и города являются ключевыми рынками?

6. Как меняется объем продаж со временем?
   Есть ли сезонность или периоды роста/падения?

7. Какие месяцы являются самыми прибыльными по продажам?

8. Кто основные клиенты компании?
   (по количеству заказов и сумме покупок)

9. Какие клиенты являются VIP?
   (RFM-анализ)

10. Есть ли проблемы с удержанием клиентов?
    (когортный анализ)

11. Какие клиентские сегменты приносят больше всего выручки?

12. Какие категории популярны у разных клиентских сегментов?

13. Какие регионы предпочитают разные категории товаров?

14. Какой способ доставки используется чаще всего?

15. Есть ли зависимость между частотой покупок клиента и общей суммой покупок?
''')


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
print('\nЧисловые признаки:')
print(df.describe())
print('\nРаспределения:')
print(df.shape)

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
print(df['category'].value_counts(normalize = True))




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

#3  Feature enginering
print("\n" + "="*70)
print("СОЗДАНИЕ НОВЫХ ПРИЗНАКОВ")
print("="*70)

#3.1 создание дат
df['year'] = df['order_date'].dt.year
df['month'] = df['order_date'].dt.month
df['day'] = df['order_date'].dt.day
df['weekday'] = df['order_date'].dt.dayofweek      # 0=пн ... 6=вс
df['week'] = df['order_date'].dt.isocalendar().week
df['quarter'] = df['order_date'].dt.quarter
df['month_name'] = df['order_date'].dt.strftime('%B')
df['weekday_name'] = df['order_date'].dt.strftime('%A')
df['is_weekend'] = df['weekday'].isin([5,6])
df['shipping_days'] = df['ship_date'] - df['order_date']

#дни спустя
df['days_since_first'] = (df['order_date'] - df['order_date'].min()).dt.days
df['days_for_last'] = ( df['order_date'].max() - df['order_date']).dt.days
#периоды



#3.2 создание метрик
customer_stats = df.groupby('customer_id').agg(
    frequency = ('order_id','count'),
    monetary = ('sales','sum'),
    recency = ('order_date','max'))
customer_stats['recency_days'] = (df['order_date'].max() - customer_stats['recency']).dt.days

#3.3 cоздание сегментов
df['sales_segment'] = pd.qcut(df['sales'],q = 4, labels = ['low','mid','high','premium'])
customer_stats['monetary_segment'] = pd.qcut(customer_stats['monetary'],q = 4, labels = ['low','mid','high','premium'])
customer_stats['frequency_segment'] = pd.qcut(customer_stats['frequency'],q = 4, labels = ['low','mid','high','premium'])
customer_stats['recency_segment'] = pd.qcut(customer_stats['recency_days'],q = 4, labels = ['premium','high','mid','low'])

customer_stats = customer_stats.reset_index()
df = df.merge(customer_stats, on = 'customer_id', how = 'left')


#4 EDA


#4.1 одномерный анализ
print("\n" + "="*50)
print('AНАЛИЗИРОВАНИЕ ПРИЗНАКОВ ОДНОМЕРНЫХ')
print("="*50)

#АНАЛИЗ ЧИСЛОВЫХ ПРИЗНАКОВ И НОВЫХ ПРИЗНАКОВ
print(df['sales'].describe())
fig, axes = plt.subplots(2,2,figsize = (14,16))
sns.boxplot(data = df[['sales','recency_days','monetary','frequency']],ax = axes[0,0])
axes[0,0].set_title('выбросы ящик с усами ')


sns.histplot(data = df, x ='recency_days',ax = axes[0,1],
                        kde = True)
axes[0,1].set_title('распределение дней с последним заказом')
axes[0,1].set_ylabel('количество клиентов')
axes[0,1].set_xlabel('дней спустя')

sns.histplot(data = df, x ='monetary',ax = axes[1,0],
                    kde = True)
axes[1,0].set_title('сколько денег приносит клиент')
axes[1,0].set_ylabel('сумма покупок')
axes[1,0].set_xlabel('количество клиентов')

sns.histplot(data = df,x = 'frequency',ax = axes[1,1],
                     kde = True)
axes[1,1].set_title('частота покупок ')
axes[1,1].set_ylabel('покупатели количество')
axes[1,1].set_xlabel('количесвто заказов')
plt.tight_layout()
plt.plot()

#ОБЩЕЕ РАСПРЕДЕЛЕНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ ПО ПРОЦЕНТАМ

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
#3
print('\nРаспределение cегментов по цене :',df['sales_segment'].value_counts())
procent_of_sales_segment_count = df['sales_segment'].value_counts() / len(df) * 100
print('\nКоличественное распределение сегментов цены по процентам:',procent_of_sales_segment_count)


#визуализация признаков
fig, axes = plt.subplots(3,3,figsize = (20,30))

#города и регионы
sns.countplot(data = df, y =  'city',
              order = df['city'].value_counts().head(10).index, ax = axes[0,0])
axes[0,0].set_title('топ 10 городов по количеству ')
axes[0,0].set_ylabel('города')
axes[0,0].set_xlabel('количество заказов')

#штатов 
sns.countplot(data = df, y =  'state',
              order = df['state'].value_counts().head(10).index, ax = axes[0,1])
axes[0,1].set_title('топ 10 штатов по количеству ')
axes[0,1].set_ylabel('штаты')
axes[0,1].set_xlabel('количество заказов')

#регионов
sns.countplot(data = df, y =  'region',
              order = df['region'].value_counts().index, ax = axes[0,2])
axes[0,2].set_title('распределение регионов')
axes[0,2].set_ylabel('регионы')
axes[0,2].set_xlabel('количество заказов')

#различные категории

#сегментов
df['segment'].value_counts().plot(kind = 'pie', autopct ='%1.1f%%',ax = axes[1,0])
axes[1,0].set_title('распределение сегментов')

#категорий
df['category'].value_counts().plot(kind = 'pie', autopct ='%1.1f%%',ax = axes[1,1])
axes[1,1].set_title('распределение категорий')

#под_категорий
sns.countplot(data = df, y =  'sub_category',
              order = df['sub_category'].value_counts().index, ax = axes[1,2])
axes[1,2].set_title('распределение под категорий ')
axes[1,2].set_ylabel('под_категории')
axes[1,2].set_xlabel('количество заказов')

#ship_mode
sns.countplot(data = df, y =  'ship_mode',
              order = df['ship_mode'].value_counts().index, ax = axes[2,0])
axes[2,0].set_title('распределение типов заказа')
axes[2,0].set_ylabel('типы заказов')
axes[2,0].set_xlabel('количество заказов')

#распределение клиентов
sns.countplot(data = df, y =  'customer_name',
              order = df['customer_name'].value_counts().head(10).index, ax = axes[2,1])
axes[2,1].set_title('топ 10 клиентов по заказам')
axes[2,1].set_ylabel('клиенты имя')
axes[2,1].set_xlabel('количество заказов')

#распределение сегментов по ценам
sns.countplot(data = df, y =  'sales_segment',
              order = df['sales_segment'].value_counts().head(10).index, ax = axes[2,2])
axes[2,2].set_title('денежные сегменты')
axes[2,2].set_ylabel('название сегментов')
axes[2,2].set_xlabel('цены')

plt.tight_layout()
plt.show()







#4.2 двумерный анализ
print("\n" + "="*50)
print('AНАЛИЗИРОВАНИЕ ПРИЗНАКОВ ДВУМЕРНЫХ')
print("="*50)



fig, axes = plt.subplots(3,3,figsize = (12,16))
#признаки категория + продажа
sales_category = df.groupby('category')['sales'].sum()
sales_subcategory = df.groupby('sub_category')['sales'].sum()
sales_region = df.groupby('region')['sales'].sum()
sales_state =df.groupby('state')['sales'].sum()
sales_city =df.groupby('city')['sales'].sum()
sales_segment = df.groupby('sales_segment')['sales'].sum()
#категория + категоия
crosstab_segment_ship_mode = pd.crosstab(df['sales_segment'],df['ship_mode'])
crosstab_region_category = pd.crosstab(df['region'],df['category'])
crosstab_category_segment = pd.crosstab(df['category'],df['sales_segment'])


print('\nЦены по категориям:', sales_subcategory,
      '\nЦены по подкатегориям:', sales_category
      )
print('\nЦены по месту:', sales_city,
     '\nЦены по месту:',sales_region,
     '\nЦены по месту:', sales_state
      )
print('\nЦены по сегментации:' ,sales_segment)


print('\nКросс таблица по сегменту и типу доставки:',crosstab_segment_ship_mode)
print('\nКросс таблица по категории и региону:',crosstab_region_category)
print('\nКросс таблица по сегменту и категории:',crosstab_category_segment)
      


#визуализация
sales_category.plot(kind = 'bar',
                    ax =axes[0,0])
sales_subcategory.plot(kind = 'bar',
                       ax =axes[0,1])
sales_region.plot(kind = 'bar',
                  ax =axes[0,2])
sales_state.plot(kind = 'bar' ,
                 ax =axes[1,0])
sales_city.plot(kind = 'bar',
                ax =axes[1,1])
sales_category.plot(kind = 'bar' ,
                    ax =axes[1,2])
crosstab_segment_ship_mode.plot(kind = 'bar',
                                ax = axes[2,0])
crosstab_region_category.plot(kind = 'bar',
                                ax = axes[2,1])
crosstab_category_segment.plot(kind = 'bar',
                                ax = axes[2,2])


plt.title('анализ старых признаков')
plt.show()

#признаки через pivot_table
pivot_category_date = df.pivot_table(
    index = 'month',
    columns = 'category',
    values = 'sales',
    aggfunc = 'sum')

pivot_region_date = df.pivot_table(
    index = 'month',
    columns = 'region',
    values = 'sales',
    aggfunc = 'sum')

pivot_segment_date = df.pivot_table(
    index = 'month',
    columns = 'segment',
    values = 'sales',
    aggfunc = 'sum')



# корреляционная матрица и визуализация
numeric_columns = df[['sales','recency_days','frequency','monetary']]
fig, axes = plt.subplots(2,2,figsize = (10,12))

sns.heatmap(pivot_category_date,annot = True ,fmt = '.0f',ax = axes[0,0])
sns.heatmap(pivot_region_date,annot = True ,fmt = '.0f',ax = axes[0,1])
sns.heatmap(pivot_segment_date,annot = True ,fmt = '.0f',ax = axes[1,0])
sns.heatmap(numeric_columns.corr(), annot = True,fmt = '0.1f',ax = axes[1,1])
plt.show()








#временной анализ









#4.3 многомерный анализ
print("\n" + "="*50)
print('AНАЛИЗИРОВАНИЕ ПРИЗНАКОВ МНОГОМЕРНЫХ')
print("="*50)

#процентное соотношение


procent_of_category_sales = df.groupby('category')['sales'].sum() / df['sales'].sum() * 100
procent_of_subcategory_sales = df.groupby('sub_category')['sales'].sum() / df['sales'].sum() * 100



#sum() aggfunction sales
total_sales_category_by_regions = df.groupby(['category','region','state','city'])['sales'].sum()
total_sales_category_by_segments =  df.groupby(['category','segment'])['sales'].sum()
total_sales_regions_by_segments = df.groupby(['segment','region','state','city'])['sales'].sum()
                                         
total_sales_subcategory_by_regions = df.groupby(['sub_category','region','state','city'])['sales'].sum()
total_sales_subcategory_by_segments = df.groupby(['sub_category','segment'])['sales'].sum()
total_sales_by_all_category = df.groupby(['category','segment'])['sales'].sum()


#mean() aggfunction sales
                                   
sales_category_by_regions = df.groupby(['category','region','state','city'])['sales'].mean()
sales_category_by_segments =  df.groupby(['category','segment'])['sales'].mean()
sales_regions_by_segments = df.groupby(['segment','region','state','city'])['sales'].mean()
                                         
sales_subcategory_by_regions = df.groupby(['sub_category','region','state','city'])['sales'].mean()
sales_subcategory_by_segments = df.groupby(['category','segment'])['sales'].mean()
sales_by_all_category = df.groupby(['category','segment'])['sales'].mean()

#новых признаков
rfm_category = df.groupby('category')[['recency_days','frequency','monetary']].mean()
rfm_region = df.groupby('region')[['recency_days','frequency','monetary']].mean()
rfm_sub_category = df.groupby('sub_category')[['recency_days','frequency','monetary']].mean()
rfm_sales_segment = df.groupby('sales_segment')[['recency_days','frequency','monetary']].mean()
rfm_ship_mode = df.groupby('ship_mode')[['recency_days','frequency','monetary']].mean()



#4.4 когортный анализ
df['cohort_date'] = df.groupby('customer_id')['order_date'].transform('min')
#месяцы
df['month_order'] = df['order_date'].dt.to_period('M')
df['month_cohort'] = df['cohort_date'].dt.to_period('M')
df['period'] = (df['month_order'] - df['month_cohort']).apply(lambda x: x.n)
#уникал пользователи
cohort = (df.groupby(['month_cohort','period'])['customer_id'].nunique().reset_index())
#таблица когорт
retention = cohort.pivot(
    index = 'month_cohort',
    columns ='period',
    values = 'customer_id')
retention_rate = retention.div(retention[0], axis = 0) * 100
print(retention_rate)

sns.heatmap(retention_rate,annot = True,fmt = '0.1f',cmap = 'Blues')
plt.title('сohort_retention')
plt.xlabel('месяц после первой покупки')
plt.ylabel('кошортная группа')
plt.show()























































