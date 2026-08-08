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


# ============================================================
# 4. EDA — EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n" + "="*70)
print("4. EDA — ИССЛЕДОВАТЕЛЬСКИЙ АНАЛИЗ")
print("="*70)


# ============================================================
# 4.1 ОБЩАЯ СТАТИСТИКА
# ============================================================

print("\n--- Общая статистика ---")

print(
    df[
        [
            'sales',
            'quantity',
            'frequency',
            'monetary',
            'recency_days',
            'shipping_days'
        ]
    ].describe()
)


# ============================================================
# 4.2 РАСПРЕДЕЛЕНИЯ ОСНОВНЫХ ЧИСЛОВЫХ ПРИЗНАКОВ
# ============================================================

numeric_features = [
    'sales',
    'quantity',
    'frequency',
    'monetary',
    'recency_days',
    'shipping_days'
]

for col in numeric_features:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x=col,
        kde=True
    )

    plt.title(f'Распределение {col}')
    plt.xlabel(col)
    plt.ylabel('Количество')

    plt.tight_layout()
    plt.show()


# ============================================================
# 4.3 ПРОДАЖИ ПО КАТЕГОРИЯМ
# ============================================================

category_sales = (
    df.groupby('category', observed=True)['sales']
    .sum()
    .sort_values(ascending=False)
)

print("\n--- Выручка по категориям ---")
print(category_sales)


plt.figure(figsize=(8, 5))

sns.barplot(
    x=category_sales.index,
    y=category_sales.values
)

plt.title('Выручка по категориям')
plt.xlabel('Категория')
plt.ylabel('Выручка')

plt.tight_layout()
plt.show()


# ============================================================
# 4.4 ПРОДАЖИ ПО ПОДКАТЕГОРИЯМ
# ============================================================

subcategory_sales = (
    df.groupby('sub_category', observed=True)['sales']
    .sum()
    .sort_values(ascending=False)
)

print("\n--- Топ-10 подкатегорий ---")
print(subcategory_sales.head(10))


plt.figure(figsize=(10, 6))

sns.barplot(
    x=subcategory_sales.head(10).values,
    y=subcategory_sales.head(10).index
)

plt.title('Топ-10 подкатегорий по выручке')
plt.xlabel('Выручка')
plt.ylabel('Подкатегория')

plt.tight_layout()
plt.show()


# ============================================================
# 4.5 ГЕОГРАФИЯ
# ============================================================

region_sales = (
    df.groupby('region')['sales']
    .sum()
    .sort_values(ascending=False)
)

state_sales = (
    df.groupby('state')['sales']
    .sum()
    .sort_values(ascending=False)
)

city_sales = (
    df.groupby('city')['sales']
    .sum()
    .sort_values(ascending=False)
)


print("\n--- Выручка по регионам ---")
print(region_sales)

print("\n--- Топ-10 штатов ---")
print(state_sales.head(10))

print("\n--- Топ-10 городов ---")
print(city_sales.head(10))


# Регионы
plt.figure(figsize=(8, 5))

sns.barplot(
    x=region_sales.index,
    y=region_sales.values
)

plt.title('Выручка по регионам')
plt.xlabel('Регион')
plt.ylabel('Выручка')

plt.tight_layout()
plt.show()


# Штаты
plt.figure(figsize=(10, 6))

sns.barplot(
    x=state_sales.head(10).values,
    y=state_sales.head(10).index
)

plt.title('Топ-10 штатов по выручке')
plt.xlabel('Выручка')
plt.ylabel('Штат')

plt.tight_layout()
plt.show()


# ============================================================
# 4.6 АНАЛИЗ КЛИЕНТОВ
# ============================================================

customer_revenue = (
    df.groupby('customer_id')['sales']
    .sum()
    .sort_values(ascending=False)
)

customer_orders = (
    df.groupby('customer_id')['order_id']
    .nunique()
    .sort_values(ascending=False)
)


print("\n--- Топ-10 клиентов по выручке ---")
print(customer_revenue.head(10))

print("\n--- Топ-10 клиентов по количеству заказов ---")
print(customer_orders.head(10))


# ============================================================
# 4.7 RFM-СЕГМЕНТАЦИЯ
# ============================================================

rfm_segment_revenue = (
    df.groupby('Customer_Segment', observed=True)['sales']
    .sum()
    .sort_values(ascending=False)
)

rfm_segment_count = (
    df.groupby('Customer_Segment', observed=True)['customer_id']
    .nunique()
    .sort_values(ascending=False)
)


print("\n--- Выручка RFM-сегментов ---")
print(rfm_segment_revenue)

print("\n--- Количество клиентов в RFM-сегментах ---")
print(rfm_segment_count)


plt.figure(figsize=(9, 5))

sns.barplot(
    x=rfm_segment_revenue.index,
    y=rfm_segment_revenue.values
)

plt.title('Выручка по RFM-сегментам')
plt.xlabel('RFM-сегмент')
plt.ylabel('Выручка')

plt.tight_layout()
plt.show()


# ============================================================
# 4.8 КАТЕГОРИЯ × RFM-СЕГМЕНТ
# ============================================================

segment_category_sales = (
    df.groupby(
        ['Customer_Segment', 'category'],
        observed=True
    )['sales']
    .sum()
    .reset_index()
)


plt.figure(figsize=(11, 6))

sns.barplot(
    data=segment_category_sales,
    x='Customer_Segment',
    y='sales',
    hue='category'
)

plt.title('Выручка категорий по RFM-сегментам')
plt.xlabel('RFM-сегмент')
plt.ylabel('Выручка')

plt.tight_layout()
plt.show()


# ============================================================
# 4.9 РЕГИОН × КАТЕГОРИЯ
# ============================================================

region_category_sales = pd.pivot_table(
    df,
    index='region',
    columns='category',
    values='sales',
    aggfunc='sum',
    observed=True
)

print("\n--- Выручка категорий по регионам ---")
print(region_category_sales)


plt.figure(figsize=(10, 6))

sns.heatmap(
    region_category_sales,
    annot=True,
    fmt='.0f'
)

plt.title('Выручка: регион × категория')
plt.xlabel('Категория')
plt.ylabel('Регион')

plt.tight_layout()
plt.show()


# ============================================================
# 4.10 FREQUENCY × MONETARY
# ============================================================

customer_rfm = (
    df[
        [
            'customer_id',
            'frequency',
            'monetary',
            'recency_days'
        ]
    ]
    .drop_duplicates('customer_id')
)


frequency_monetary_corr = (
    customer_rfm['frequency']
    .corr(customer_rfm['monetary'])
)


print(
    f"\nКорреляция Frequency × Monetary: "
    f"{frequency_monetary_corr:.3f}"
)


plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=customer_rfm,
    x='frequency',
    y='monetary'
)

plt.title('Frequency × Monetary')
plt.xlabel('Frequency')
plt.ylabel('Monetary')

plt.tight_layout()
plt.show()

#5. СТАТИСТИЧЕСКИЙ ТЕСТЫ
print("\n" + "="*50)
print('СТАТИСТИЧЕСКИЕ ТЕСТЫ')
print("="*50)

#5.1 Проверка различий продаж между сегментами клиентов


from scipy.stats import kruskal

segments = [
    group['sales'].values 
    for name, group in df.groupby('segment')
]

stat, p_value = kruskal(*segments)

print('Kruskal-Wallis тест')
print('Статистика:', stat)
print('p-value:', p_value)

if p_value < 0.05:
    print('Есть статистически значимые различия между сегментами')
else:
    print('Статистически значимых различий нет')




#5.2 Проверка связи между частотой покупок и выручкой клиента

from scipy.stats import spearmanr

stat, p_value = spearmanr(
    customer_stats['frequency'],
    customer_stats['monetary']
)

print('Spearman correlation:', stat)
print('p-value:', p_value)

if p_value < 0.05:
    print('Связь статистически значимая')
else:
    print('Связь не доказана')
    
#5.3 Проверка роста продаж по годам


year_sales = (
    df.groupby('year')['sales']
    .sum()
    .reset_index()
)

stat, p_value = spearmanr(
    year_sales['year'],
    year_sales['sales']
)

print('Тренд продаж')
print('Корреляция:', stat)
print('p-value:', p_value)

#5.4 Проверка зависимости категории и региона

from scipy.stats import chi2_contingency

table = pd.crosstab(
    df['region'],
    df['category']
)

chi2, p_value, dof, expected = chi2_contingency(table)

print('Chi-square test')
print('p-value:', p_value)

if p_value < 0.05:
    print('Категория товара зависит от региона')
else:
    print('Зависимость не доказана')

    
#5.5 Проверка различий времени доставки


shipping_groups = [
    group['shipping_days'].dt.days.values
    for name, group in df.groupby('ship_mode')
]

stat, p_value = kruskal(*shipping_groups)

print('Тест времени доставки')
print('p-value:', p_value)



# ============================================================
# 6. ОТВЕТЫ НА БИЗНЕС-ВОПРОСЫ
# ============================================================

print("\n" + "="*70)
print("6. ОТВЕТЫ НА БИЗНЕС-ВОПРОСЫ")
print("="*70)


# ============================================================
# 1. КАКИЕ КАТЕГОРИИ ПРИНОСЯТ МАКСИМАЛЬНУЮ ВЫРУЧКУ?
# ============================================================

top_category = category_sales.idxmax()
top_category_revenue = category_sales.max()

print(
    f"\n1. Лидирующая категория: {top_category}"
    f"\n   Выручка: ${top_category_revenue:,.2f}"
)


# ============================================================
# 2. КАКИЕ ПОДКАТЕГОРИИ ЯВЛЯЮТСЯ ЛИДЕРАМИ?
# ============================================================

print("\n2. Топ-5 подкатегорий:")

print(
    subcategory_sales.head(5)
)


# ============================================================
# 3. КАКИЕ ТОВАРЫ ИМЕЮТ САМЫЙ ВЫСОКИЙ СПРОС?
# ============================================================

product_demand = (
    df.groupby('product_name')['order_id']
    .nunique()
    .sort_values(ascending=False)
)

print("\n3. Топ-10 товаров по количеству заказов:")

print(
    product_demand.head(10)
)


# ============================================================
# 4. КАКИЕ РЕГИОНЫ ПРИНОСЯТ БОЛЬШЕ ВСЕГО ВЫРУЧКИ?
# ============================================================

print("\n4. Выручка по регионам:")

print(
    region_sales
)


# ============================================================
# 5. КАКИЕ ШТАТЫ И ГОРОДА ЯВЛЯЮТСЯ КЛЮЧЕВЫМИ?
# ============================================================

print("\n5. Топ-5 штатов:")

print(
    state_sales.head(5)
)

print("\nТоп-5 городов:")

print(
    city_sales.head(5)
)


# ============================================================
# 6. КАК МЕНЯЕТСЯ ОБЪЁМ ПРОДАЖ СО ВРЕМЕНЕМ?
# ============================================================

monthly_sales = (
    df.set_index('order_date')
    .resample('ME')['sales']
    .sum()
)


monthly_change = (
    monthly_sales
    .pct_change()
    .mul(100)
)


print("\n6. Продажи по месяцам:")

print(
    monthly_sales
)


print("\nИзменение относительно предыдущего месяца (%):")

print(
    monthly_change.round(2)
)


# Тренд
plt.figure(figsize=(12, 5))

sns.lineplot(
    x=monthly_sales.index,
    y=monthly_sales.values,
    marker='o'
)

plt.title('Динамика выручки по месяцам')
plt.xlabel('Месяц')
plt.ylabel('Выручка')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ============================================================
# 7. КАКИЕ МЕСЯЦЫ САМЫЕ СИЛЬНЫЕ?
# ============================================================

print("\n7. Топ-5 месяцев по выручке:")

print(
    monthly_sales
    .sort_values(ascending=False)
    .head(5)
)


# ============================================================
# 8. КТО ОСНОВНЫЕ КЛИЕНТЫ?
# ============================================================

print("\n8. Топ-10 клиентов по количеству заказов:")

print(
    customer_orders.head(10)
)


print("\nТоп-10 клиентов по выручке:")

print(
    customer_revenue.head(10)
)


# ============================================================
# 9. КАКИЕ КЛИЕНТЫ VIP?
# ============================================================

vip_customers = (
    customer_stats[
        customer_stats['monetary_segment'].eq('premium') &
        customer_stats['frequency_segment'].eq('premium') &
        customer_stats['recency_segment'].eq('premium')
    ]
    .sort_values('monetary', ascending=False)
)


print("\n9. VIP-клиенты:")

print(
    vip_customers[
        [
            'customer_id',
            'recency_days',
            'frequency',
            'monetary'
        ]
    ].head(20)
)


# ============================================================
# 10. ЕСТЬ ЛИ ПРОБЛЕМЫ С УДЕРЖАНИЕМ?
# ============================================================

df['order_month'] = (
    df['order_date']
    .dt.to_period('M')
)


df['cohort_month'] = (
    df.groupby('customer_id')['order_date']
    .transform('min')
    .dt.to_period('M')
)


df['cohort_index'] = (
    (df['order_month'].dt.year - df['cohort_month'].dt.year) * 12
    +
    (df['order_month'].dt.month - df['cohort_month'].dt.month)
)


cohort_data = (
    df.groupby(
        ['cohort_month', 'cohort_index']
    )['customer_id']
    .nunique()
    .reset_index()
)


cohort_table = cohort_data.pivot(
    index='cohort_month',
    columns='cohort_index',
    values='customer_id'
)


retention = (
    cohort_table
    .div(cohort_table[0], axis=0)
    * 100
)


print("\n10. Retention:")

print(
    retention.round(1)
)


plt.figure(figsize=(12, 7))

sns.heatmap(
    retention,
    annot=True,
    fmt='.1f'
)

plt.title('Cohort Retention (%)')
plt.xlabel('Месяц после первой покупки')
plt.ylabel('Когорта')

plt.tight_layout()
plt.show()


# ============================================================
# 11. КАКИЕ КЛИЕНТСКИЕ СЕГМЕНТЫ ПРИНОСЯТ БОЛЬШЕ ВСЕГО?
# ============================================================

print("\n11. Выручка RFM-сегментов:")

print(
    rfm_segment_revenue
)


print("\nДоля выручки RFM-сегментов (%):")

print(
    (
        rfm_segment_revenue
        / rfm_segment_revenue.sum()
        * 100
    ).round(2)
)


# ============================================================
# 12. КАКИЕ КАТЕГОРИИ ПОПУЛЯРНЫ У РАЗНЫХ СЕГМЕНТОВ?
# ============================================================

print("\n12. Категории по RFM-сегментам:")

print(
    segment_category_sales
    .sort_values(
        ['Customer_Segment', 'sales'],
        ascending=[True, False]
    )
)


# ============================================================
# 13. КАКИЕ РЕГИОНЫ ПРЕДПОЧИТАЮТ РАЗНЫЕ КАТЕГОРИИ?
# ============================================================

print("\n13. Регион × категория:")

print(
    region_category_sales
)


# ============================================================
# 14. КАКОЙ СПОСОБ ДОСТАВКИ ИСПОЛЬЗУЕТСЯ ЧАЩЕ?
# ============================================================

ship_mode_usage = (
    df.groupby('ship_mode')['order_id']
    .nunique()
    .sort_values(ascending=False)
)


ship_mode_share = (
    ship_mode_usage
    / ship_mode_usage.sum()
    * 100
)


print("\n14. Использование способов доставки:")

print(
    ship_mode_usage
)


print("\nДоля способов доставки (%):")

print(
    ship_mode_share.round(2)
)


# ============================================================
# 15. FREQUENCY × MONETARY
# ============================================================

correlation = (
    customer_rfm['frequency']
    .corr(customer_rfm['monetary'])
)


print(
    f"\n15. Корреляция Frequency × Monetary: "
    f"{correlation:.3f}"
)


# ============================================================
# 7. BUSINESS INSIGHTS & FINAL CONCLUSIONS
# ============================================================

print("\n" + "="*70)
print("7. BUSINESS INSIGHTS & FINAL CONCLUSIONS")
print("="*70)


# ============================================================
# 7.1 ОСНОВНЫЕ БИЗНЕС-МЕТРИКИ
# ============================================================

total_revenue = df['sales'].sum()
total_orders = df['order_id'].nunique()
total_customers = df['customer_id'].nunique()

avg_order_value = total_revenue / total_orders
avg_customer_revenue = total_revenue / total_customers


print("\n--- КЛЮЧЕВЫЕ МЕТРИКИ ---")

print(f"Общая выручка: ${total_revenue:,.2f}")
print(f"Количество заказов: {total_orders:,}")
print(f"Количество клиентов: {total_customers:,}")
print(f"Средний чек: ${avg_order_value:,.2f}")
print(f"Средняя выручка на клиента: ${avg_customer_revenue:,.2f}")


# ============================================================
# 7.2 ЛИДЕРЫ
# ============================================================

top_category = category_sales.idxmax()
top_subcategory = subcategory_sales.idxmax()
top_region = region_sales.idxmax()
top_state = state_sales.idxmax()
top_city = city_sales.idxmax()

print("\n--- ЛИДЕРЫ ---")

print(f"Лучшая категория: {top_category}")
print(f"Лучшая подкатегория: {top_subcategory}")
print(f"Лучший регион: {top_region}")
print(f"Лучший штат: {top_state}")
print(f"Лучший город: {top_city}")


# ============================================================
# 7.3 КОНЦЕНТРАЦИЯ ВЫРУЧКИ
# ============================================================

top_category_share = (
    category_sales.iloc[0] /
    category_sales.sum()
    * 100
)

top_region_share = (
    region_sales.iloc[0] /
    region_sales.sum()
    * 100
)

top_10_customer_share = (
    customer_revenue.head(10).sum()
    / customer_revenue.sum()
    * 100
)

print("\n--- КОНЦЕНТРАЦИЯ ВЫРУЧКИ ---")

print(
    f"Доля крупнейшей категории: "
    f"{top_category_share:.2f}%"
)

print(
    f"Доля крупнейшего региона: "
    f"{top_region_share:.2f}%"
)

print(
    f"Доля топ-10 клиентов: "
    f"{top_10_customer_share:.2f}%"
)


# ============================================================
# 7.4 ДИНАМИКА
# ============================================================

best_month = monthly_sales.idxmax()
worst_month = monthly_sales.idxmin()

best_month_sales = monthly_sales.max()
worst_month_sales = monthly_sales.min()

best_month_change = monthly_change.idxmax()
worst_month_change = monthly_change.idxmin()

print("\n--- ВРЕМЕННАЯ ДИНАМИКА ---")

print(
    f"Лучший месяц: {best_month.strftime('%Y-%m')} "
    f"(${best_month_sales:,.2f})"
)

print(
    f"Худший месяц: {worst_month.strftime('%Y-%m')} "
    f"(${worst_month_sales:,.2f})"
)

print(
    f"Максимальный месячный рост: "
    f"{best_month_change.strftime('%Y-%m')} "
    f"({monthly_change.loc[best_month_change]:.2f}%)"
)

print(
    f"Максимальное месячное падение: "
    f"{worst_month_change.strftime('%Y-%m')} "
    f"({monthly_change.loc[worst_month_change]:.2f}%)"
)


# ============================================================
# 7.5 КЛИЕНТСКАЯ СТРУКТУРА
# ============================================================

segment_share = (
    rfm_segment_revenue
    / rfm_segment_revenue.sum()
    * 100
)

largest_segment = rfm_segment_revenue.idxmax()

print("\n--- КЛИЕНТСКИЕ СЕГМЕНТЫ ---")

print(
    f"Крупнейший сегмент по выручке: "
    f"{largest_segment}"
)

print("\nДоля выручки по сегментам:")

print(
    segment_share.round(2)
)


# ============================================================
# 7.6 FREQUENCY × MONETARY
# ============================================================

print("\n--- FREQUENCY × MONETARY ---")

print(
    f"Корреляция: "
    f"{correlation:.3f}"
)

if correlation >= 0.7:
    print(
        "Наблюдается сильная положительная связь: "
        "более частые покупки обычно связаны "
        "с большей общей выручкой клиента."
    )

elif correlation >= 0.4:
    print(
        "Наблюдается умеренная положительная связь."
    )

else:
    print(
        "Связь слабая или отсутствует."
    )


# ============================================================
# 7.7 RETENTION
# ============================================================

if 1 in retention.columns:

    retention_m1 = retention[1].mean()

    print("\n--- RETENTION ---")

    print(
        f"Средний retention на второй месяц: "
        f"{retention_m1:.2f}%"
    )


# ============================================================
# 7.8 ФИНАЛЬНЫЕ НАБЛЮДЕНИЯ
# ============================================================

print("\n" + "="*70)
print("КЛЮЧЕВЫЕ НАБЛЮДЕНИЯ")
print("="*70)

print(f"""
1. {top_category} является крупнейшей категорией
   по общей выручке.

2. {top_region} является крупнейшим регионом
   по выручке.

3. {top_subcategory} является лидирующей
   подкатегорией.

4. Наибольший объём выручки приходится
   на сегмент: {largest_segment}.

5. Топ-10 клиентов формируют
   {top_10_customer_share:.2f}% общей выручки.

6. Максимальная выручка наблюдается
   в {best_month.strftime('%Y-%m')}.

7. Минимальная выручка наблюдается
   в {worst_month.strftime('%Y-%m')}.

8. Корреляция Frequency × Monetary:
   {correlation:.3f}.
""")


# ============================================================
# 7.9 БИЗНЕС-ГИПОТЕЗЫ ДЛЯ ДАЛЬНЕЙШЕЙ ПРОВЕРКИ
# ============================================================

print("\n" + "="*70)
print("ГИПОТЕЗЫ ДЛЯ ДОПОЛНИТЕЛЬНОЙ ПРОВЕРКИ")
print("="*70)

print("""
1. Проверить, какие товары формируют выручку
   крупнейшего клиентского сегмента.

2. Проверить сезонность отдельных категорий
   и их вклад в месячные падения/рост.

3. Проверить, концентрируется ли выручка
   вокруг небольшого количества клиентов.

4. Проверить, какие категории чаще покупают
   клиенты с высокой Frequency.

5. Проверить причины низкого Retention:
   категории, повторные покупки, регионы,
   каналы и время первой покупки.

6. Проверить прибыльность, если в данных
   появится себестоимость или profit.
""")


# ============================================================
# 7.10 ФИНАЛЬНАЯ РЕКОМЕНДАЦИЯ
# ============================================================

print("\n" + "="*70)
print("РЕКОМЕНДАЦИИ")
print("="*70)

print("""
1. Сфокусировать дальнейший анализ на категориях
   и сегментах, формирующих основную выручку.

2. Исследовать причины месячных падений продаж
   через категории, товары и регионы.

3. Отдельно работать с высокоценных клиентов,
   поскольку их поведение может существенно
   влиять на общую выручку.

4. Изучить категории с высокой частотой покупок
   для поиска возможностей повторных продаж.

5. Для анализа прибыльности добавить Profit,
   Cost и Discount, если эти данные доступны.

6. Не делать причинные выводы только на основе
   корреляций и наблюдений. Причины необходимо
   подтверждать дополнительными проверками.
""")










