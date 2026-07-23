import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

# Настройки отображения
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', '{:.2f}'.format)

print("Текущая директория:", os.getcwd())

# ============================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================
df = pd.read_csv(r'C:\Users\User\Downloads\netflix_titles.csv.zip')

print("\n" + "="*50)
print("ПЕРВИЧНЫЙ ОСМОТР ДАННЫХ")
print("="*50)
print(df.head())
print("\nИнформация о датафрейме:")
print(df.info())
print("\nТипы данных:")
print(df.dtypes)
print("\nСтатистика по числовым колонкам:")
print(df.describe())

# ============================================
# 2. ОЧИСТКА ДАННЫХ
# ============================================
print("\n" + "="*50)
print("ОЧИСТКА ДАННЫХ")
print("="*50)

# Приводим текстовые колонки к единому формату
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.strip().str.lower()

# Преобразуем дату добавления
df['date_added'] = pd.to_datetime(df['date_added'], dayfirst=True, errors='coerce')

# Заполняем пропуски
df['director'] = df['director'].fillna('неизвестно')
df['cast'] = df['cast'].fillna('неизвестно')
df['country'] = df['country'].fillna('неизвестно')
df['rating'] = df['rating'].fillna('no rating')

# Удаляем записи без даты добавления (их мало, но они портят анализ)
df = df.dropna(subset=['date_added'])

print(f"Пропуски после очистки:\n{df.isna().sum()}")
print(f"\nКоличество уникальных значений:\n{df.nunique()}")

# Проверяем, что остались только корректные даты
missing_dates = df[df['date_added'].isna()]
print(f"\nЗаписей с некорректной датой: {len(missing_dates)}")

# Оставляем только первую страну из списка (для упрощения анализа)
df['country'] = df['country'].str.split(',').str[0]

# ============================================
# 3. АНАЛИЗ КОНТЕНТА ПО СТРАНАМ
# ============================================
print("\n" + "="*50)
print("АНАЛИЗ ПО СТРАНАМ")
print("="*50)

# Популярность типов контента
popular_type = df['type'].value_counts()
print(f"Распределение по типам:\n{popular_type}")

# Топ-10 стран-производителей
top_10_countries = df.groupby('country')['show_id'].count().sort_values(ascending=False).head(10)
print(f"\nТоп-10 стран по количеству контента:\n{top_10_countries}")

# Доля топ-10 стран
top10_percent = (top_10_countries.sum() / len(df)) * 100
print(f"\nТоп-10 стран производят {top10_percent:.1f}% всего контента")

# ============================================
# 4. ВРЕМЕННОЙ АНАЛИЗ
# ============================================
print("\n" + "="*50)
print("ВРЕМЕННОЙ АНАЛИЗ")
print("="*50)

# Извлекаем компоненты даты
df['year'] = df['date_added'].dt.year
df['month'] = df['date_added'].dt.month
df['weekday'] = df['date_added'].dt.dayofweek  # 0 = понедельник

# Контент по годам
content_by_year = df.groupby('year')['show_id'].count().reset_index()
content_by_year.columns = ['year', 'count']
print(f"Контент по годам:\n{content_by_year.head()}")

# Контент по месяцам (все года вместе)
content_by_month = df.groupby('month')['show_id'].count().reset_index()
content_by_month.columns = ['month', 'count']
print(f"\nКонтент по месяцам:\n{content_by_month}")

# Контент по дням недели
content_by_weekday = df.groupby('weekday')['show_id'].count().reset_index()
content_by_weekday.columns = ['weekday', 'count']
days = ['пн', 'вт', 'ср', 'чт', 'пт', 'сб', 'вс']
content_by_weekday['day_name'] = content_by_weekday['weekday'].map(dict(enumerate(days)))
print(f"\nКонтент по дням недели:\n{content_by_weekday}")

# ============================================
# 5. АНАЛИЗ РЕЙТИНГОВ
# ============================================
print("\n" + "="*50)
print("АНАЛИЗ РЕЙТИНГОВ")
print("="*50)

# Самый популярный рейтинг
rating_counts = df['rating'].value_counts()
top_rating = rating_counts.idxmax()
top_rating_count = rating_counts.max()
top_rating_percent = (top_rating_count / len(df)) * 100

print(f"Самый популярный рейтинг: '{top_rating}'")
print(f"Количество: {top_rating_count} ({top_rating_percent:.1f}% от всех записей)")
print(f"\nВсе рейтинги:\n{rating_counts.head(10)}")

# ============================================
# 6. СЕГМЕНТАЦИЯ КОНТЕНТА
# ============================================
print("\n" + "="*50)
print("СЕГМЕНТАЦИЯ КОНТЕНТА")
print("="*50)

# 6.1 По году выпуска
def content_age(year):
    if year >= 2020:
        return 'new'
    elif year >= 2010:
        return 'medium'
    else:
        return 'old'

df['content_age'] = df['release_year'].apply(content_age)
print(f"Сегментация по году выпуска:\n{df['content_age'].value_counts()}")

# 6.2 По возрастной категории
def age_segment(rating):
    if rating in ['g', 'tv-y', 'tv-y7']:
        return 'kids'
    elif rating in ['pg', 'pg-13', 'tv-g', 'tv-pg', 'tv-14']:
        return 'teen'
    elif rating in ['r', 'tv-ma', 'nc-17']:
        return 'adult'
    else:
        return 'unknown'

df['age_category'] = df['rating'].apply(age_segment)
print(f"\nСегментация по возрастной категории:\n{df['age_category'].value_counts()}")

# 6.3 По длительности для фильмов
movies = df[df['type'] == 'movie'].copy()
movies['duration_min'] = movies['duration'].str.extract('(\d+)').astype(float)

def movie_length_segment(minutes):
    if minutes < 90:
        return 'short'
    elif minutes <= 120:
        return 'standard'
    else:
        return 'long'

movies['length_segment'] = movies['duration_min'].apply(movie_length_segment)
print(f"\nСегментация фильмов по длительности:\n{movies['length_segment'].value_counts()}")

# 6.4 По длительности для сериалов
shows = df[df['type'] == 'tv show'].copy()
shows['duration_seasons'] = shows['duration'].str.extract('(\d+)').astype(float)

def show_length_segment(seasons):
    if seasons == 1:
        return 'short'
    elif seasons < 3:
        return 'standard'
    elif seasons < 5:
        return 'long'
    else:
        return 'very_long'

shows['length_segment'] = shows['duration_seasons'].apply(show_length_segment)
print(f"\nСегментация сериалов по количеству сезонов:\n{shows['length_segment'].value_counts()}")

# 6.5 По активности стран
df['content_count'] = df.groupby('country')['country'].transform('count')
df['country_activity'] = pd.cut(
    df['content_count'],
    bins=[0, 100, 500, float('inf')],
    labels=['low', 'mid', 'high']
)
print(f"\nАктивность стран по производству контента:\n{df['country_activity'].value_counts()}")

# ============================================
# 7. ВИЗУАЛИЗАЦИЯ
# ============================================
print("\n" + "="*50)
print("ВИЗУАЛИЗАЦИЯ")
print("="*50)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Распределение по типам
df['type'].value_counts().plot(kind='bar', ax=axes[0, 0], color=['skyblue', 'lightcoral'])
axes[0, 0].set_title('Типы контента')
axes[0, 0].set_xlabel('Тип')
axes[0, 0].set_ylabel('Количество')

# 2. Топ-10 стран
top_10_countries.plot(kind='bar', ax=axes[0, 1], color='lightgreen')
axes[0, 1].set_title('Топ-10 стран по контенту')
axes[0, 1].set_xlabel('Страна')
axes[0, 1].set_ylabel('Количество')

# 3. Контент по годам
content_by_year.plot(x='year', y='count', kind='line', ax=axes[0, 2], marker='o')
axes[0, 2].set_title('Динамика добавления контента')
axes[0, 2].set_xlabel('Год')
axes[0, 2].set_ylabel('Количество')

# 4. Контент по месяцам
content_by_month.plot(x='month', y='count', kind='bar', ax=axes[1, 0], color='orange')
axes[1, 0].set_title('Сезонность по месяцам')
axes[1, 0].set_xlabel('Месяц')
axes[1, 0].set_ylabel('Количество')

# 5. Контент по дням недели
content_by_weekday.plot(x='day_name', y='count', kind='bar', ax=axes[1, 1], color='purple')
axes[1, 1].set_title('Активность по дням недели')
axes[1, 1].set_xlabel('День недели')
axes[1, 1].set_ylabel('Количество')

# 6. Возрастные категории
df['age_category'].value_counts().plot(kind='bar', ax=axes[1, 2], color='teal')
axes[1, 2].set_title('Возрастные категории')
axes[1, 2].set_xlabel('Категория')
axes[1, 2].set_ylabel('Количество')

plt.tight_layout()
plt.show()

# ============================================
# 8. АНАЛИЗ ВЫБРОСОВ ПО ДЛИТЕЛЬНОСТИ ФИЛЬМОВ
# ============================================
print("\n" + "="*50)
print("АНАЛИЗ ВЫБРОСОВ")
print("="*50)

print(f"Статистика длительности фильмов:\n{movies['duration_min'].describe()}")

# Определяем выбросы через IQR
q1 = movies['duration_min'].quantile(0.25)
q3 = movies['duration_min'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = movies[
    (movies['duration_min'] < lower_bound) |
    (movies['duration_min'] > upper_bound)
]

print(f"\nКоличество фильмов-выбросов по длительности: {len(outliers)}")
print(f"Границы: от {lower_bound:.0f} до {upper_bound:.0f} минут")
print(f"\nПримеры выбросов (первые 5):")
print(outliers[['title', 'duration_min']].head())

# ============================================
# 9. ИТОГОВАЯ СТАТИСТИКА
# ============================================
print("\n" + "="*50)
print("ИТОГОВАЯ СТАТИСТИКА")
print("="*50)

print(f"Всего записей: {len(df)}")
print(f"Фильмов: {len(movies)}")
print(f"Сериалов: {len(shows)}")
print(f"Уникальных стран: {df['country'].nunique()}")
print(f"Диапазон дат: с {df['date_added'].min().date()} по {df['date_added'].max().date()}")
print(f"Самый популярный рейтинг: {top_rating}")

# ============================================
# 10. ВЫВОДЫ
# ============================================
print("\n" + "="*50)
print("КЛЮЧЕВЫЕ ИНСАЙТЫ")
print("="*50)

print("""
1. Большая часть контента на Netflix приходится на фильмы
2. США доминируют в производстве контента
3. Пик добавления контента приходится на 2018-2020 годы
4. Самый популярный возрастной рейтинг - "TV-MA" (для взрослых)
5. Большинство фильмов имеют стандартную длительность (90-120 минут)
6. Сериалы чаще всего имеют 1-2 сезона
7. Наблюдается сезонность: больше контента добавляют в конце года
""")




















