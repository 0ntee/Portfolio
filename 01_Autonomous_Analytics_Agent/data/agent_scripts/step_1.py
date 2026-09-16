import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
df = pd.read_csv('data/ecommerce_transactions.csv')

# Проверка данных на пропуски
print('Пропуски в Age: ', df['Age'].isnull().sum())
print('Пропуски в Purchase_Amount: ', df['Purchase_Amount'].isnull().sum())

# Заполнение пропусков медианными значениями
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Purchase_Amount'] = df['Purchase_Amount'].fillna(df['Purchase_Amount'].median())

# Разбивка на возрастные группы
age_bins = [0, 30, 55, 100]
age_labels = ['Молодежь', 'Средний возраст', 'Пожилые']
df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

# Посчитать средний чек для каждой группы
average_spending = df.groupby('Age_Group')['Purchase_Amount'].mean().reset_index()
print('Средний чек для каждой группы:')
print(average_spending)

# Построение графиков
sns.kdeplot(data=df, x='Purchase_Amount', hue='Age_Group', alpha=0.7, legend=True)
plt.xlabel('Средний чек')
plt.ylabel('Количество')
plt.title('Плотность распределения трат по возрастным группам')
plt.savefig('data/outputs/age_spending_distribution.png', bbox_inches='tight')
plt.close()