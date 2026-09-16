import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных
df = pd.read_csv('data/ecommerce_transactions.csv')

# Проверка наличия пропусков и заполнение медианными значениями
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Purchase_Amount'].fillna(df['Purchase_Amount'].median(), inplace=True)

# Разбиение на возрастные группы
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 55, 100], labels=['Молодежь', 'Средний возраст', 'Пожилые'])

# Посчитать средний чек для каждой группы
grouped_data = df.groupby('Age_Group')['Purchase_Amount'].mean().reset_index()

# Вывести средние чеки
print("Средний чек для каждой группы:\n", grouped_data)

# Построить KDE plot для каждой группы
plt.figure(figsize=(12, 6))
sns.kdeplot(grouped_data['Purchase_Amount'], hue='Age_Group', alpha=0.7, palette='viridis')
plt.title('Диаграмма плотности распределения трат по возрастным группам')
plt.xlabel('Средний чек')
plt.ylabel('Функция распределения')
plt.savefig('data/outputs/age_spending_distribution.png', bbox_inches='tight')
plt.close()