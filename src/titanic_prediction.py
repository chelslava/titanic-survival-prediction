"""
Модуль для предсказания выживаемости пассажиров Титаника.
Включает в себя полный пайплайн: загрузку данных, предобработку, обучение моделей,
оценку качества и сохранение лучшей модели.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import joblib
from typing import Tuple, Dict, Any

# Машинное обучение
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

# Метрики оценки
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

# Отключаем предупреждения для чистоты вывода
warnings.filterwarnings('ignore')

class TitanicPredictor:
    """
    Класс для предсказания выживаемости пассажиров Титаника.
    Включает методы для предобработки данных, обучения моделей и оценки качества.
    """
    
    def __init__(self):
        """Инициализация предиктора"""
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.features = []
    
    def load_data(self, filepath: str = './data/train.csv') -> pd.DataFrame:
        """
        Загружает данные из CSV файла
        
        Args:
            filepath: путь к файлу с данными
            
        Returns:
            DataFrame с загруженными данными
        """
        try:
            self.df = pd.read_csv(filepath)
            print(f"✅ Данные успешно загружены из {filepath}")
            print(f"📊 Размер датасета: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            print(f"❌ Ошибка: файл {filepath} не найден")
            return None
    
    def explore_data(self) -> None:
        """Выполняет базовый исследовательский анализ данных"""
        if self.df is None:
            print("❌ Данные не загружены. Сначала используйте load_data()")
            return
        
        print("=" * 50)
        print("📊 ИССЛЕДОВАТЕЛЬСКИЙ АНАЛИЗ ДАННЫХ")
        print("=" * 50)
        
        # Основная информация о датасете
        print("\n🔍 Основная информация:")
        print(self.df.info())
        
        # Статистическое описание
        print("\n📈 Описательная статистика:")
        print(self.df.describe())
        
        # Пропущенные значения
        print("\n🕳️ Пропущенные значения:")
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Количество': missing_data,
            'Процент': missing_percent
        })
        print(missing_df[missing_df['Количество'] > 0])
        
        # Распределение целевой переменной
        print("\n🎯 Распределение целевой переменной (Survived):")
        survived_counts = self.df['Survived'].value_counts()
        print(f"Не выжило: {survived_counts[0]} ({survived_counts[0]/len(self.df)*100:.1f}%)")
        print(f"Выжило: {survived_counts[1]} ({survived_counts[1]/len(self.df)*100:.1f}%)")
    
    def visualize_data(self) -> None:
        """Создает визуализации для анализа данных"""
        if self.df is None:
            print("❌ Данные не загружены")
            return
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Анализ выживаемости пассажиров Титаника', fontsize=16)
        
        # 1. Распределение выживших
        sns.countplot(data=self.df, x='Survived', ax=axes[0, 0])
        axes[0, 0].set_title('Распределение выживших')
        axes[0, 0].set_xlabel('Выжил (0 = Нет, 1 = Да)')
        
        # 2. Выживаемость по полу
        sns.barplot(data=self.df, x='Sex', y='Survived', ax=axes[0, 1])
        axes[0, 1].set_title('Выживаемость по полу')
        
        # 3. Выживаемость по классу
        sns.barplot(data=self.df, x='Pclass', y='Survived', ax=axes[0, 2])
        axes[0, 2].set_title('Выживаемость по классу')
        
        # 4. Распределение возраста
        sns.histplot(data=self.df, x='Age', hue='Survived', kde=True, ax=axes[1, 0])
        axes[1, 0].set_title('Распределение возраста')
        
        # 5. Выживаемость по классу и полу
        sns.barplot(data=self.df, x='Pclass', y='Survived', hue='Sex', ax=axes[1, 1])
        axes[1, 1].set_title('Выживаемость по классу и полу')
        
        # 6. Распределение стоимости билетов
        sns.histplot(data=self.df, x='Fare', hue='Survived', kde=True, ax=axes[1, 2])
        axes[1, 2].set_title('Распределение стоимости билетов')
        
        plt.tight_layout()
        plt.show()
    
    def preprocess_data(self) -> None:
        """Выполняет предобработку данных"""
        if self.df is None:
            print("❌ Данные не загружены")
            return
        
        print("🔧 Начинаем предобработку данных...")
        
        # Заполнение пропущенных значений
        print("  • Заполнение пропущенных значений возраста медианой")
        self.df['Age'] = self.df['Age'].fillna(self.df['Age'].median())
        
        print("  • Заполнение пропущенных значений порта посадки модой")
        self.df['Embarked'] = self.df['Embarked'].fillna(self.df['Embarked'].mode()[0])
        
        print("  • Заполнение пропущенных значений стоимости билета медианой")
        self.df['Fare'] = self.df['Fare'].fillna(self.df['Fare'].median())
        
        # Кодирование категориальных признаков
        print("  • Кодирование пола: male -> 0, female -> 1")
        self.df['Sex'] = self.df['Sex'].map({'male': 0, 'female': 1})
        
        print("  • One-hot кодирование порта посадки")
        self.df = pd.get_dummies(self.df, columns=['Embarked'], drop_first=True)
        
        # Создание новых признаков
        print("  • Создание признака размера семьи")
        self.df['FamilySize'] = self.df['SibSp'] + self.df['Parch']
        
        print("  • Создание признака одиночества")
        self.df['IsAlone'] = (self.df['FamilySize'] == 0).astype(int)
        
        print("  • Извлечение титулов из имен")
        self.df['Title'] = self.df['Name'].str.extract(r',\s*([^\.]+)\.', expand=False)
        
        # Группировка редких титулов
        title_counts = self.df['Title'].value_counts()
        rare_titles = title_counts[title_counts < 10].index
        self.df['Title'] = self.df['Title'].replace(rare_titles, 'Rare')
        
        # Унификация титулов
        title_mapping = {
            'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
            'Lady': 'Royalty', 'Countess': 'Royalty', 'Dona': 'Royalty',
            'Capt': 'Officer', 'Col': 'Officer', 'Major': 'Officer',
            'Dr': 'Officer', 'Rev': 'Officer',
            'Don': 'Noble', 'Sir': 'Noble', 'Jonkheer': 'Noble'
        }
        self.df['Title'] = self.df['Title'].replace(title_mapping)
        
        # One-hot кодирование титулов
        self.df = pd.get_dummies(self.df, columns=['Title'], drop_first=True)
        
        print("  • Создание возрастных групп")
        self.df['AgeGroup'] = pd.cut(
            self.df['Age'], 
            bins=[0, 12, 18, 35, 60, 100], 
            labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior']
        )
        self.df = pd.get_dummies(self.df, columns=['AgeGroup'], drop_first=True)
        
        print("  • Создание признака наличия каюты")
        self.df['HasCabin'] = self.df['Cabin'].notnull().astype(int)
        
        # Удаление неинформативных колонок
        print("  • Удаление неинформативных признаков")
        self.df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
        
        # Формирование списка признаков
        title_features = [col for col in self.df.columns if col.startswith('Title_')]
        agegroup_features = [col for col in self.df.columns if col.startswith('AgeGroup_')]
        embarked_features = [col for col in self.df.columns if col.startswith('Embarked_')]
        
        self.features = [
            'Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'HasCabin'
        ] + title_features + agegroup_features + embarked_features
        
        print(f"✅ Предобработка завершена. Итоговое количество признаков: {len(self.features)}")
    
    def prepare_train_test(self, test_size: float = 0.2, random_state: int = 42) -> None:
        """
        Подготавливает данные для обучения и тестирования
        
        Args:
            test_size: размер тестовой выборки
            random_state: зерно случайности
        """
        if self.df is None:
            print("❌ Данные не обработаны")
            return
        
        # Выделение признаков и целевой переменной
        X = self.df[self.features]
        y = self.df['Survived']
        
        # Разделение на обучающую и тестовую выборки
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"📊 Данные разделены:")
        print(f"  • Обучающая выборка: {self.X_train.shape}")
        print(f"  • Тестовая выборка: {self.X_test.shape}")
    
    def train_models(self) -> Dict[str, Any]:
        """
        Обучает несколько моделей машинного обучения
        
        Returns:
            Словарь с обученными моделями
        """
        if self.X_train is None:
            print("❌ Данные не подготовлены для обучения")
            return {}
        
        print("🤖 Начинаем обучение моделей...")
        
        models = {}
        
        # 1. Логистическая регрессия
        print("  • Обучение логистической регрессии")
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(self.X_train, self.y_train)
        models['LogisticRegression'] = lr
        
        # 2. Случайный лес
        print("  • Обучение случайного леса")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.X_train, self.y_train)
        models['RandomForest'] = rf
        
        # 3. XGBoost
        print("  • Обучение XGBoost")
        xgb = XGBClassifier(eval_metric='logloss', random_state=42)
        xgb.fit(self.X_train, self.y_train)
        models['XGBoost'] = xgb
        
        print("✅ Все модели успешно обучены")
        return models
    
    def evaluate_models(self, models: Dict[str, Any]) -> pd.DataFrame:
        """
        Оценивает качество обученных моделей
        
        Args:
            models: словарь с обученными моделями
            
        Returns:
            DataFrame с метриками качества
        """
        if not models or self.X_test is None:
            print("❌ Модели не обучены или данные не подготовлены")
            return pd.DataFrame()
        
        print("📊 Оценка качества моделей...")
        
        results = []
        
        for name, model in models.items():
            # Предсказания
            y_pred = model.predict(self.X_test)
            y_prob = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Метрики
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_prob) if y_prob is not None else None
            
            results.append({
                'Модель': name,
                'Accuracy': f"{accuracy:.4f}",
                'Precision': f"{precision:.4f}",
                'Recall': f"{recall:.4f}",
                'F1-Score': f"{f1:.4f}",
                'ROC-AUC': f"{roc_auc:.4f}" if roc_auc else "N/A"
            })
            
            print(f"  • {name}: Accuracy = {accuracy:.4f}")
        
        results_df = pd.DataFrame(results)
        return results_df
    
    def optimize_models(self) -> Dict[str, Any]:
        """
        Выполняет подбор гиперпараметров для моделей
        
        Returns:
            Словарь с оптимизированными моделями
        """
        if self.X_train is None:
            print("❌ Данные не подготовлены")
            return {}
        
        print("⚙️ Оптимизация гиперпараметров...")
        
        optimized_models = {}
        
        # Логистическая регрессия
        print("  • Оптимизация логистической регрессии")
        param_grid_lr = {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear']
        }
        
        grid_lr = GridSearchCV(
            LogisticRegression(max_iter=1000, random_state=42),
            param_grid_lr, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_lr.fit(self.X_train, self.y_train)
        optimized_models['LogisticRegression'] = grid_lr.best_estimator_
        print(f"    Лучшие параметры: {grid_lr.best_params_}")
        
        # Случайный лес
        print("  • Оптимизация случайного леса")
        param_grid_rf = {
            'n_estimators': [100, 200],
            'max_depth': [4, 6, 8, None],
            'min_samples_split': [2, 5, 10]
        }
        
        grid_rf = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_rf.fit(self.X_train, self.y_train)
        optimized_models['RandomForest'] = grid_rf.best_estimator_
        print(f"    Лучшие параметры: {grid_rf.best_params_}")
        
        # XGBoost
        print("  • Оптимизация XGBoost")
        param_grid_xgb = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        
        grid_xgb = GridSearchCV(
            XGBClassifier(eval_metric='logloss', random_state=42),
            param_grid_xgb, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_xgb.fit(self.X_train, self.y_train)
        optimized_models['XGBoost'] = grid_xgb.best_estimator_
        print(f"    Лучшие параметры: {grid_xgb.best_params_}")
        
        print("✅ Оптимизация завершена")
        return optimized_models
    
    def create_ensemble(self, models: Dict[str, Any]) -> VotingClassifier:
        """
        Создает ансамбль из лучших моделей
        
        Args:
            models: словарь с моделями
            
        Returns:
            Обученный ансамбль моделей
        """
        if not models or self.X_train is None:
            print("❌ Модели не готовы или данные не подготовлены")
            return None
        
        print("🎭 Создание ансамбля моделей...")
        
        # Создание ансамбля
        estimators = [(name.lower(), model) for name, model in models.items()]
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft'  # Используем вероятности для голосования
        )
        
        # Обучение ансамбля
        ensemble.fit(self.X_train, self.y_train)
        
        print("✅ Ансамбль успешно создан и обучен")
        return ensemble
    
    def evaluate_final_model(self, model: Any) -> None:
        """
        Детальная оценка финальной модели
        
        Args:
            model: обученная модель для оценки
        """
        if model is None or self.X_test is None:
            print("❌ Модель не готова или данные не подготовлены")
            return
        
        print("🏆 ФИНАЛЬНАЯ ОЦЕНКА МОДЕЛИ")
        print("=" * 40)
        
        # Предсказания
        y_pred = model.predict(self.X_test)
        y_prob = model.predict_proba(self.X_test)[:, 1]
        
        # Основные метрики
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_prob)
        
        print(f"📊 Метрики качества:")
        print(f"  • Accuracy:  {accuracy:.4f}")
        print(f"  • Precision: {precision:.4f}")
        print(f"  • Recall:    {recall:.4f}")
        print(f"  • F1-Score:  {f1:.4f}")
        print(f"  • ROC-AUC:   {roc_auc:.4f}")
        
        # Матрица ошибок
        print(f"\n🔍 Матрица ошибок:")
        cm = confusion_matrix(self.y_test, y_pred)
        print("Предсказано:   Не выжил  Выжил")
        print(f"Не выжил         {cm[0,0]:6d}   {cm[0,1]:5d}")
        print(f"Выжил            {cm[1,0]:6d}   {cm[1,1]:5d}")
        
        # Детальный отчет
        print(f"\n📄 Детальный отчет:")
        print(classification_report(self.y_test, y_pred, 
                                  target_names=['Не выжил', 'Выжил']))
        
        self.best_model = model
    
    def save_model(self, model: Any, filepath: str = '../models/titanic_model.pkl') -> None:
        """
        Сохраняет обученную модель
        
        Args:
            model: модель для сохранения
            filepath: путь для сохранения
        """
        try:
            joblib.dump(model, filepath)
            print(f"✅ Модель успешно сохранена в {filepath}")
        except Exception as e:
            print(f"❌ Ошибка при сохранении модели: {e}")
    
    def run_full_pipeline(self, data_path: str = '../data/train.csv') -> None:
        """
        Запускает полный пайплайн обучения
        
        Args:
            data_path: путь к данным
        """
        print("🚀 ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА МАШИННОГО ОБУЧЕНИЯ")
        print("=" * 60)
        
        # 1. Загрузка данных
        self.load_data(data_path)
        if self.df is None:
            return
        
        # 2. Исследовательский анализ
        self.explore_data()
        
        # 3. Визуализация (опционально)
        self.visualize_data()
        
        # 4. Предобработка
        self.preprocess_data()
        
        # 5. Подготовка данных
        self.prepare_train_test()
        
        # 6. Обучение базовых моделей
        basic_models = self.train_models()
        
        # 7. Оценка базовых моделей
        basic_results = self.evaluate_models(basic_models)
        print("\n📊 Результаты базовых моделей:")
        print(basic_results.to_string(index=False))
        
        # 8. Оптимизация моделей
        optimized_models = self.optimize_models()
        
        # 9. Оценка оптимизированных моделей
        optimized_results = self.evaluate_models(optimized_models)
        print("\n📊 Результаты оптимизированных моделей:")
        print(optimized_results.to_string(index=False))
        
        # 10. Создание ансамбля
        ensemble = self.create_ensemble(optimized_models)
        
        # 11. Финальная оценка
        self.evaluate_final_model(ensemble)
        
        # 12. Сохранение модели
        self.save_model(ensemble)
        
        print("\n🎉 ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН!")


def main():
    """Основная функция для запуска предиктора"""
    # Создание экземпляра предиктора
    predictor = TitanicPredictor()
    
    # Запуск полного пайплайна
    predictor.run_full_pipeline('../data/train.csv')


if __name__ == "__main__":
    main()