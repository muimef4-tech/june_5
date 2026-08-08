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

# ============================================================
# 4. EDA (Exploratory Data Analysis)
# ============================================================

print("\n" + "="*60)
print("EDA - ИССЛЕДОВАТЕЛЬСКИЙ АНАЛИЗ")
print("="*60)


# ============================================================
# 4.1 ОБЩИЙ АНАЛИЗ ЧИСЛОВЫХ ПРИЗНАКОВ
# ============================================================

print("\nОПИСАТЕЛЬНАЯ СТАТИСТИКА")

print(df[['sales','frequency','monetary','recency_days']].describe())


# распределение числовых признаков

fig, axes = plt.subplots(2,2, figsize=(14,10))


sns.histplot(
    data=df,
    x='sales',
    kde=True,
    ax=axes[0,0]
)
axes[0,0].set_title('Распределение продаж')


sns.histplot(
    data=df,
    x='frequency',
    kde=True,
    ax=axes[0,1]
)
axes[0,1].set_title('Частота покупок клиентов')


sns.histplot(
    data=df,
    x='monetary',
    kde=True,
    ax=axes[1,0]
)
axes[1,0].set_title('Общая сумма покупок клиентов')


sns.histplot(
    data=df,
    x='recency_days',
    kde=True,
    ax=axes[1,1]
)
axes[1,1].set_title('Дней с последней покупки')


plt.tight_layout()
plt.show()



# ============================================================
# 4.2 АНАЛИЗ ПРОДАЖ ПО КАТЕГОРИЯМ
# ============================================================


sales_category = (
    df.groupby('category')['sales']
    .sum()
    .sort_values(ascending=False)
)


sales_subcategory = (
    df.groupby('sub_category')['sales']
    .sum()
    .sort_values(ascending=False)
)


print("\nПродажи по категориям:")
print(sales_category)


print("\nПродажи по подкатегориям:")
print(sales_subcategory)



fig, axes = plt.subplots(1,2, figsize=(14,5))


sales_category.plot(
    kind='bar',
    ax=axes[0]
)

axes[0].set_title(
    'Выручка по категориям'
)


sales_subcategory.plot(
    kind='bar',
    ax=axes[1]
)

axes[1].set_title(
    'Выручка по подкатегориям'
)


plt.tight_layout()
plt.show()



# ============================================================
# 4.3 ГЕОГРАФИЧЕСКИЙ АНАЛИЗ
# ============================================================


sales_region = (
    df.groupby('region')['sales']
    .sum()
    .sort_values(ascending=False)
)


sales_state = (
    df.groupby('state')['sales']
    .sum()
    .sort_values(ascending=False)
)


sales_city = (
    df.groupby('city')['sales']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)


print("\nПродажи по регионам")
print(sales_region)


fig, axes = plt.subplots(1,3, figsize=(18,6))


sales_region.plot(
    kind='bar',
    ax=axes[0]
)

axes[0].set_title(
    'Продажи по регионам'
)



sales_state.head(10).plot(
    kind='bar',
    ax=axes[1]
)

axes[1].set_title(
    'Топ штатов'
)



sales_city.plot(
    kind='bar',
    ax=axes[2]
)

axes[2].set_title(
    'Топ городов'
)


plt.tight_layout()
plt.show()



# ============================================================
# 4.4 АНАЛИЗ КЛИЕНТОВ (RFM)
# ============================================================


print("\nТОП КЛИЕНТОВ ПО ВЫРУЧКЕ")

print(
    customer_stats
    .sort_values(
        'monetary',
        ascending=False
    )
    .head(10)
)



segment_sales = (
    df.groupby('segment')['sales']
    .sum()
    .sort_values(ascending=False)
)


print("\nПродажи по клиентским сегментам")
print(segment_sales)



segment_sales.plot(
    kind='bar',
    figsize=(8,5),
    title='Выручка по сегментам'
)

plt.ylabel('Sales')
plt.show()



# ============================================================
# 4.5 ВРЕМЕННОЙ АНАЛИЗ
# ============================================================


monthly_sales = (
    df.groupby(
        ['year','month']
    )['sales']
    .sum()
    .reset_index()
)


monthly_sales['date'] = pd.to_datetime(
    monthly_sales['year'].astype(str)
    + '-'
    + monthly_sales['month'].astype(str)
)



plt.figure(figsize=(12,5))

sns.lineplot(
    data=monthly_sales,
    x='date',
    y='sales',
    marker='o'
)


plt.title(
    'Продажи по месяцам'
)

plt.xticks(rotation=45)

plt.show()



year_sales = (
    df.groupby('year')['sales']
    .sum()
)


year_sales.plot(
    kind='bar',
    title='Продажи по годам',
    figsize=(8,5)
)

plt.show()



# ============================================================
# 4.6 АНАЛИЗ СВЯЗЕЙ
# ============================================================


numeric = df[
    [
        'sales',
        'frequency',
        'monetary',
        'recency_days'
    ]
]


plt.figure(figsize=(8,6))

sns.heatmap(
    numeric.corr(),
    annot=True,
    fmt='.2f'
)


plt.title(
    'Корреляционная матрица'
)

plt.show()



# ============================================================
# 4.7 PIVOT ANALYSIS
# ============================================================


pivot_category_date = df.pivot_table(
    index='month',
    columns='category',
    values='sales',
    aggfunc='sum'
)


plt.figure(figsize=(10,6))

sns.heatmap(
    pivot_category_date,
    annot=True,
    fmt='.0f'
)

plt.title(
    'Продажи категорий по месяцам'
)

plt.show()



# ============================================================
# 4.8 КОГОРТНЫЙ АНАЛИЗ
# ============================================================


df['cohort_date'] = (
    df.groupby('customer_id')['order_date']
    .transform('min')
)


df['month_order'] = (
    df['order_date']
    .dt.to_period('M')
)


df['month_cohort'] = (
    df['cohort_date']
    .dt.to_period('M')
)


df['period'] = (
    df['month_order']
    -
    df['month_cohort']
).apply(lambda x:x.n)



cohort = (
    df.groupby(
        ['month_cohort','period']
    )['customer_id']
    .nunique()
    .reset_index()
)


retention = cohort.pivot(
    index='month_cohort',
    columns='period',
    values='customer_id'
)


retention_rate = (
    retention
    .div(
        retention[0],
        axis=0
    )
    *100
)


# последние когорты





plt.figure(figsize=(12,7))

sns.heatmap(
    retention_rate,
    annot= False,
    fmt='.1f'
)


plt.title(
    'Cohort Retention'
)

plt.xlabel(
    'Месяц после первой покупки'
)

plt.ylabel(
    'Когорта'
)


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



#6 ОТВЕТЫ НА БИЗНЕС ВОПРОСЫ


# 1. КАКИЕ КАТЕГОРИИ ПРИНОСЯТ МАКСИМАЛЬНУЮ ВЫРУЧКУ?

category_sales = (
    df.groupby("category")["sales"]
    .sum()
    .sort_values(ascending=False)
)

top_category = category_sales.idxmax()
top_category_sales = category_sales.max()

print(
    f"\n1. Лидирующая категория: {top_category}"
    f"\nВыручка: ${top_category_sales:,.2f}"
)

print("\nВыручка по категориям:")
print(category_sales)



# 2. КАКИЕ ПОДКАТЕГОРИИ ЯВЛЯЮТСЯ ЛИДЕРАМИ ПО ПРОДАЖАМ?


subcategory_sales = (
    df.groupby("sub_category")["sales"]
    .sum()
    .sort_values(ascending=False)
)

top_subcategories = subcategory_sales.head(5)

print("\n2. Топ-5 подкатегорий по выручке:")
print(top_subcategories)

print(
    f"\nОбщая выручка топ-5 подкатегорий: "
    f"${top_subcategories.sum():,.2f}"
)



# 3. КАКИЕ ТОВАРЫ ИМЕЮТ САМЫЙ ВЫСОКИЙ СПРОС?


# Количество заказов по товарам
product_demand = (
    df.groupby("product_name")["order_id"]
    .nunique()
    .sort_values(ascending=False)
)

top_products = product_demand.head(10)

print("\n3. Топ-10 товаров по количеству заказов:")
print(top_products)


# Спрос по подкатегориям
subcategory_demand = (
    df.groupby("sub_category")["order_id"]
    .nunique()
    .sort_values(ascending=False)
)

print("\nТоп-10 подкатегорий по количеству заказов:")
print(subcategory_demand.head(10))



# 4. КАКИЕ РЕГИОНЫ ПРИНОСЯТ БОЛЬШЕ ВСЕГО ВЫРУЧКИ?


# Города
city_sales = (
    df.groupby("city")["sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\n4. Топ-10 городов по выручке:")
print(city_sales.head(10))


# Штаты
state_sales = (
    df.groupby("state")["sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nТоп-10 штатов по выручке:")
print(state_sales.head(10))


# Регионы
region_sales = (
    df.groupby("region")["sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nВыручка по регионам:")
print(region_sales)



# 5. КАКИЕ ШТАТЫ И ГОРОДА ЯВЛЯЮТСЯ КЛЮЧЕВЫМИ РЫНКАМИ?

top_state = state_sales.idxmax()
top_state_sales = state_sales.max()

top_city = city_sales.idxmax()
top_city_sales = city_sales.max()

print(
    f"\n5. Ключевой штат: {top_state}"
    f"\nВыручка: ${top_state_sales:,.2f}"
)

print(
    f"\nКлючевой город: {top_city}"
    f"\nВыручка: ${top_city_sales:,.2f}"
)


# =========================================================
# 6. КАК МЕНЯЕТСЯ ОБЪЁМ ПРОДАЖ СО ВРЕМЕНЕМ?
# =========================================================

monthly_sales = (
    df.set_index("order_date")
    .resample("ME")["sales"]
    .sum()
)

print("\n6. Продажи по месяцам:")
print(monthly_sales)


# Изменение относительно предыдущего месяца
monthly_change = (
    monthly_sales
    .pct_change()
    .mul(100)
)

print("\nИзменение продаж относительно предыдущего месяца (%):")
print(monthly_change.round(2))


# График продаж
plt.figure(figsize=(12, 5))

sns.lineplot(
    x=monthly_sales.index,
    y=monthly_sales.values,
    marker="o"
)

plt.title("Динамика выручки по месяцам")
plt.xlabel("Месяц")
plt.ylabel("Выручка")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# =========================================================
# 7. КАКИЕ МЕСЯЦЫ ЯВЛЯЮТСЯ САМЫМИ СИЛЬНЫМИ ПО ВЫРУЧКЕ?
# =========================================================

top_months = (
    monthly_sales
    .sort_values(ascending=False)
    .head(5)
)

print("\n7. Топ-5 месяцев по выручке:")
print(top_months)



# 8. КТО ОСНОВНЫЕ КЛИЕНТЫ КОМПАНИИ?


# По количеству заказов
customer_orders = (
    df.groupby("customer_id")["order_id"]
    .nunique()
    .sort_values(ascending=False)
)

print("\n8. Топ-10 клиентов по количеству заказов:")
print(customer_orders.head(10))


# По выручке
customer_revenue = (
    df.groupby("customer_id")["sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nТоп-10 клиентов по выручке:")
print(customer_revenue.head(10))




df["order_date"] = pd.to_datetime(df["order_date"])


# =========================================================
# 9. КАКИЕ КЛИЕНТЫ ЯВЛЯЮТСЯ VIP?
#    RFM-анализ
# =========================================================

analysis_date = df["order_date"].max() + pd.Timedelta(days=1)

rfm = (
    df.groupby("customer_id")
    .agg(
        Recency=("order_date",
                 lambda x: (analysis_date - x.max()).days),

        Frequency=("order_id", "nunique"),

        Monetary=("sales", "sum")
    )
    .reset_index()
)

print("\n9. RFM:")
print(rfm.head())


# RFM-оценки от 1 до 5
# Для Recency меньше = лучше, поэтому переворачиваем

rfm["R"] = pd.qcut(
    rfm["Recency"],
    5,
    labels=[5, 4, 3, 2, 1]
)

rfm["F"] = pd.qcut(
    rfm["Frequency"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
)

rfm["M"] = pd.qcut(
    rfm["Monetary"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
)


rfm["RFM_Score"] = (
    rfm["R"].astype(int)
    + rfm["F"].astype(int)
    + rfm["M"].astype(int)
)


# Сегментация
def segment_customer(row):

    if row["RFM_Score"] >= 13:
        return "VIP"

    elif row["RFM_Score"] >= 10:
        return "Loyal"

    elif row["RFM_Score"] >= 7:
        return "Potential"

    else:
        return "At Risk"


rfm["Customer_Segment"] = rfm.apply(
    segment_customer,
    axis=1
)


print("\nVIP-клиенты:")
print(
    rfm[rfm["Customer_Segment"] == "VIP"]
    .sort_values("Monetary", ascending=False)
)


# =========================================================
# 10. ЕСТЬ ЛИ ПРОБЛЕМЫ С УДЕРЖАНИЕМ?
#     КОГОРТНЫЙ АНАЛИЗ
# =========================================================

# Месяц заказа
df["order_month"] = df["order_date"].dt.to_period("M")


# Первая покупка клиента
df["cohort_month"] = (
    df.groupby("customer_id")["order_date"]
    .transform("min")
    .dt.to_period("M")
)


# Индекс месяца относительно первой покупки
df["cohort_index"] = (
    (df["order_month"].dt.year - df["cohort_month"].dt.year) * 12
    +
    (df["order_month"].dt.month - df["cohort_month"].dt.month)
)


# Уникальные клиенты
cohort_data = (
    df.groupby(
        ["cohort_month", "cohort_index"]
    )["customer_id"]
    .nunique()
    .reset_index()
)


# Таблица когорт
cohort_table = cohort_data.pivot(
    index="cohort_month",
    columns="cohort_index",
    values="customer_id"
)


# Retention %
retention = (
    cohort_table
    .div(cohort_table[0], axis=0)
    * 100
)


print("\n10. Retention:")
print(retention.round(1))


# Heatmap
plt.figure(figsize=(12, 6))

sns.heatmap(
    retention,
    annot=True,
    fmt=".1f"
)

plt.title("Customer Retention by Cohort")
plt.xlabel("Months Since First Purchase")
plt.ylabel("Cohort")

plt.tight_layout()
plt.show()


# =========================================================
# 11. КАКИЕ КЛИЕНТСКИЕ СЕГМЕНТЫ ПРИНОСЯТ БОЛЬШЕ ВСЕГО ВЫРУЧКИ?
# =========================================================

segment_revenue = (
    df.merge(
        rfm[["customer_id", "Customer_Segment"]],
        on="customer_id",
        how="left"
    )
    .groupby("Customer_Segment")["sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\n11. Выручка по клиентским сегментам:")
print(segment_revenue)


# Доля выручки
segment_share = (
    segment_revenue
    / segment_revenue.sum()
    * 100
)

print("\nДоля выручки:")
print(segment_share.round(2))


# =========================================================
# 12. КАКИЕ КАТЕГОРИИ ПОПУЛЯРНЫ У РАЗНЫХ СЕГМЕНТОВ?
# =========================================================

segment_category = (
    df.merge(
        rfm[["customer_id", "Customer_Segment"]],
        on="customer_id",
        how="left"
    )
    .groupby(
        ["Customer_Segment", "category"]
    )["sales"]
    .sum()
    .reset_index()
)


print("\n12. Категории по клиентским сегментам:")
print(segment_category)


# Визуализация
plt.figure(figsize=(12, 6))

sns.barplot(
    data=segment_category,
    x="Customer_Segment",
    y="sales",
    hue="category",
    estimator="sum"
)

plt.title("Sales by Customer Segment and Category")
plt.xlabel("Customer Segment")
plt.ylabel("Sales")

plt.tight_layout()
plt.show()


# =========================================================
# 13. КАКИЕ РЕГИОНЫ ПРЕДПОЧИТАЮТ РАЗНЫЕ КАТЕГОРИИ?
# =========================================================

region_category = pd.crosstab(
    df["region"],
    df["category"],
    values=df["sales"],
    aggfunc="sum"
)

print("\n13. Выручка категорий по регионам:")
print(region_category)


# Heatmap
plt.figure(figsize=(10, 6))

sns.heatmap(
    region_category,
    annot=True,
    fmt=".0f"
)

plt.title("Sales by Region and Category")
plt.xlabel("Category")
plt.ylabel("Region")

plt.tight_layout()
plt.show()


# =========================================================
# 14. КАКОЙ СПОСОБ ДОСТАВКИ ИСПОЛЬЗУЕТСЯ ЧАЩЕ?
# =========================================================

ship_mode_usage = (
    df.groupby("ship_mode")["order_id"]
    .nunique()
    .sort_values(ascending=False)
)

print("\n14. Использование способов доставки:")
print(ship_mode_usage)


# Доля каждого способа
ship_mode_share = (
    ship_mode_usage
    / ship_mode_usage.sum()
    * 100
)

print("\nДоля способов доставки:")
print(ship_mode_share.round(2))


# =========================================================
# 15. ЕСТЬ ЛИ ЗАВИСИМОСТЬ МЕЖДУ ЧАСТОТОЙ ПОКУПОК
#     И ОБЩЕЙ СУММОЙ ПОКУПОК?
# =========================================================

frequency_monetary = (
    df.groupby("customer_id")
    .agg(
        Frequency=("order_id", "nunique"),
        Monetary=("sales", "sum")
    )
    .reset_index()
)


# Корреляция
correlation = (
    frequency_monetary["Frequency"]
    .corr(frequency_monetary["Monetary"])
)

print(
    f"\n15. Корреляция Frequency и Monetary: "
    f"{correlation:.3f}"
)


# Scatterplot
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=frequency_monetary,
    x="Frequency",
    y="Monetary"
)

plt.title("Frequency vs Monetary")
plt.xlabel("Number of Orders")
plt.ylabel("Total Sales")

plt.tight_layout()
plt.show()

















































