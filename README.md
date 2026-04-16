<p align="center">
  <img src="images/app_screenshot.png" alt="Real Estate Valuation App" width="600">
</p>

# Real Estate Valuation App

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)

**Интеллектуальный сервис оценки рыночной стоимости недвижимости**

Проект решает задачу предсказания цены объекта на основе его характеристик, географического положения и данных об инфраструктуре. Включает полный цикл: от обработки «сырых» данных до готового веб-приложения на Streamlit.


## Цель проекта

Помочь риелторам и частным инвесторам:
- быстро оценивать рыночную стоимость объекта;
- сравнивать цену в объявлении с прогнозом модели;
- выявлять потенциально недооценённые или переоценённые предложения.


## Что внутри

- **`dz_bLite_final_handcrafted_App2.ipynb`** – основной ноутбук с исследованием, feature engineering, обучением и сравнением моделей.
- **`app.py`** – веб-интерфейс на Streamlit для демонстрации работы модели.
- **`requirements.txt`** – зависимости проекта.
- **`data/real_estate_sample.csv`** – 1% семпл данных для быстрого тестирования.
- **`images`** – графики и скриншоты.



## Немного цифр и графиков

### Распределение цены
Сильная асимметрия исходного распределения — одна из причин, почему мы работаем с логарифмом целевой переменной.

<p align="center">
  <img src="images/eda_main.png" alt="EDA" width="700">
</p>

### Важность признаков
Модель опирается в первую очередь на географические признаки и площадь объекта.

<p align="center">
  <img src="images/lightgbm_gain_importance.png" alt="Top важных признаков" width="500">
</p>



## Быстрый старт

1. **Клонируйте репозиторий**
   ```bash
   git clone https://github.com/kuchsk/skillfactory_realestate.git
   cd real-estate-valuation
