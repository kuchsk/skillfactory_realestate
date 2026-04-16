import os
import pickle
import re
import numpy as np
import pandas as pd
import streamlit as st

# Настройка страницы
st.set_page_config(page_title="Оценка недвижимости", layout="wide")
st.title("Интеллектуальный сервис оценки недвижимости")
st.caption("LightGBM-модель для оценки рыночной стоимости и поиска выгодных предложений")

BUNDLE_PATH = os.path.join("output", "real_estate_bundle.pkl")

# Вспомогательные функции (копия логики из ноутбука)
RE_NUM = re.compile(r"[-+]?\d*\.?\d+")

def parse_num(val):
    if val is None or pd.isna(val):
        return np.nan
    m = RE_NUM.search(str(val).replace(",", ""))
    return float(m.group()) if m else np.nan

def safe_div(x, y, fallback=np.nan):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.full(len(x), fallback, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y) | (y == 0))
    out[mask] = x[mask] / y[mask]
    return out

def prepare_lgbm_frame_final(X, num_features, cat_features, num_medians, cat_modes):
    Xp = X.copy()
    for c in num_features:
        if c not in Xp.columns:
            Xp[c] = np.nan
        Xp[c] = pd.to_numeric(Xp[c], errors="coerce").fillna(num_medians.get(c, 0))
    for c in cat_features:
        if c not in Xp.columns:
            Xp[c] = cat_modes.get(c, "unknown")
        Xp[c] = Xp[c].fillna(cat_modes.get(c, "unknown")).astype(str).astype("category")
    return Xp[num_features + cat_features]

def build_inference_frame(raw_row, bundle):
    # Формирует признаки для одного объекта по логике ноутбука
    row = dict(raw_row)

    # Базовые парсинги
    row["sqftClean"] = parse_num(row.get("sqft"))
    row["bedsClean"] = parse_num(row.get("beds"))
    row["bathsClean"] = parse_num(row.get("baths"))
    row["storiesClean"] = parse_num(row.get("stories"))

    # Тип недвижимости
    prop_type = row.get("propertyType", "other")
    if prop_type in ("singlefamily", "condo", "other"):
        row["propertyTypeNorm"] = prop_type
    else:
        row["propertyTypeNorm"] = "other"

    # Категориальные поля
    for c in ["state", "city", "zipcode", "propertyTypeNorm"]:
        val = row.get(c, "unknown")
        row[c] = "unknown" if val is None or pd.isna(val) else str(val)

    row["zipPrefix3"] = row["zipcode"][:3] if row["zipcode"] not in ("unknown", "nan") else "unk"

    # Возраст и ремонт
    ref_year = bundle["reference_year"]
    year_built = row.get("hfyearbuilt", np.nan)
    row["hfyearbuilt"] = year_built
    row["houseAge"] = ref_year - year_built if not pd.isna(year_built) else np.nan
    row["isMissingYear"] = int(pd.isna(year_built))

    remodeled_year = row.get("hfremodeledyear", np.nan)
    row["hfremodeledyear"] = remodeled_year
    row["hfyearsSinceReno"] = ref_year - remodeled_year if not pd.isna(remodeled_year) else np.nan
    row["hfwasRemodeled"] = int(not pd.isna(remodeled_year))
    row["isMissingReno"] = int(pd.isna(remodeled_year))

    row["isNewBuild"] = int((not pd.isna(row["houseAge"])) and row["houseAge"] <= 3)
    row["isHistoric"] = int((not pd.isna(row["houseAge"])) and row["houseAge"] >= 100)

    # Участок
    lot = row.get("hflotsqft", np.nan)
    row["hflotsqft"] = lot
    row["isMissingLot"] = int(pd.isna(lot))
    row["hflogLotsqft"] = np.log1p(max(lot, 0)) if not pd.isna(lot) else np.nan

    # Парковка и инфо
    parking_spaces = row.get("hfparkingspaces", np.nan)
    row["hfparkingspaces"] = parking_spaces
    row["hfparkinginfo"] = int(
        (not pd.isna(parking_spaces))
        or row.get("hfhasgarage", 0) == 1
        or row.get("hfhascarport", 0) == 1
        or row.get("hfhasstreet", 0) == 1)

    # Бассейн и другие флаги
    row["hasPoolUnified"] = int(row.get("hfhaspool", 0) == 1)
    for flag in ["hfhascentralac", "hfhasforcedair", "hfhasgarage",
                 "hfhasfireplace", "hfhascarport"]:
        row[flag] = int(row.get(flag, 0))

    # Школы
    school_avg = row.get("schoolratingavg", np.nan)
    row["schoolratingavg"] = school_avg
    row["schoolratingmax"] = row.get("schoolratingmax", np.nan)
    row["schoolratingcount"] = row.get("schoolratingcount", 0.0)
    row["schoolmindistmi"] = row.get("schoolmindistmi", np.nan)
    row["schoolhashighschool"] = int(row.get("schoolhashighschool", 0))
    row["isMissingSchoolRating"] = int(pd.isna(school_avg))
    row["isMissingSchoolDist"] = int(pd.isna(row["schoolmindistmi"]))

    # Производные признаки
    row["logSqft"] = np.log1p(max(row["sqftClean"], 0)) if not pd.isna(row["sqftClean"]) else np.nan
    row["bathPerBed"] = safe_div([row.get("bathsClean")], [row.get("bedsClean")])[0]
    row["sqftPerBed"] = safe_div([row.get("sqftClean")], [row.get("bedsClean")])[0]
    row["sqftPerBath"] = safe_div([row.get("sqftClean")], [row.get("bathsClean")])[0]
    total_rooms = (0 if pd.isna(row.get("bedsClean")) else row.get("bedsClean", 0)) \
                + (0 if pd.isna(row.get("bathsClean")) else row.get("bathsClean", 0))
    row["totalRooms"] = total_rooms
    denom = total_rooms if total_rooms != 0 else np.nan
    row["lotPerRoom"] = safe_div([lot], [denom])[0] if not pd.isna(lot) else np.nan

    # Zip-гео признаки
    zg = bundle["zip_geo_maps"]
    z = row["zipcode"]
    row["zip_med"] = float(zg["zip_med_map"].get(z, zg["zip_med_default"]))
    row["zip_pps"] = float(zg["zip_pps_map"].get(z, zg["zip_pps_default"]))
    row["zip_cntlog"] = float(np.log1p(zg["zip_cnt_map"].get(z, 0)))

    # Формируем датафрейм с нужными колонками
    X_new = pd.DataFrame([row])
    for c in bundle["feature_columns"]:
        if c not in X_new.columns:
            X_new[c] = np.nan

    # Финальная подготовка (заполнение медианами, приведение категорий)
    X_new = prepare_lgbm_frame_final(
        X_new[bundle["feature_columns"]],
        bundle["num_features"],
        bundle["cat_features"],
        bundle["num_medians"],
        bundle["cat_modes"])
    return X_new

# Загрузка модели и метаданных
@st.cache_resource
def load_bundle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    bundle = load_bundle(BUNDLE_PATH)
except FileNotFoundError:
    st.error(f"Файл модели не найден: {BUNDLE_PATH}. Сначала запустите финальный ноутбук")
    st.stop()

# Загрузка справочников для гео-подсказок
state_to_cities = bundle.get("state_to_cities", {})
city_to_zips = bundle.get("city_to_zips", {})

if not state_to_cities:
    st.error("В bundle отсутствуют словари гео-подсказок. Обновите ноутбук и пересохраните модель.")
    st.stop()

# Интерфейс: две колонки для компактного ввода

col1, col2 = st.columns(2)

with col1:
    st.subheader("Основные характеристики")
    sqft = st.number_input("Площадь (sqft)", min_value=300, max_value=20000, value=1800, step=50)
    beds = st.number_input("Спальни", min_value=1, max_value=15, value=3)
    baths = st.number_input("Ванные", min_value=1, max_value=10, value=2)
    stories = st.number_input("Этажность", min_value=1, max_value=5, value=1)
    year_built = st.number_input("Год постройки", min_value=1800, max_value=2026, value=2005)
    lot_size = st.number_input("Участок (sqft)", min_value=0, max_value=5_000_000, value=5000, step=500)
    
    st.subheader("Парковка и удобства")
    parking_spaces = st.number_input("Парковочных мест", min_value=0, max_value=10, value=1)
    has_garage = st.checkbox("Гараж", value=True)
    has_carport = st.checkbox("Навес для машины (carport)", value=False)
    has_pool = st.checkbox("Бассейн", value=False)
    has_fireplace = st.checkbox("Камин", value=True)
    has_ac = st.checkbox("Центральный кондиционер", value=True)
    has_fa = st.checkbox("Принудительное отопление", value=True)

with col2:
    st.subheader("Местоположение")
    state = st.selectbox(
        "Штат",
        options=sorted(state_to_cities.keys()),
        index=None,
        placeholder="Выберите или начните вводить...")
    if state:
        cities = state_to_cities.get(state, [])
        city = st.selectbox(
            "Город",
            options=cities,
            index=None,
            placeholder="Выберите или начните вводить..." )
    else:
        city = st.selectbox("Город", [], disabled=True)
    
    if city:
        zips = city_to_zips.get(city, [])
        zipcode = st.selectbox(
            "ZIP-код",
            options=zips,
            index=None,
            placeholder="Выберите или начните вводить...")
    else:
        zipcode = st.selectbox("ZIP-код", [], disabled=True)
    
    prop_type = st.selectbox("Тип недвижимости", ["singlefamily", "condo", "other"], index=0)
    
    st.subheader("Школы")
    school_avg = st.slider("Средний рейтинг школ", 0.0, 10.0, 6.5, 0.1)
    school_max = st.slider("Максимальный рейтинг", 0.0, 10.0, 8.0, 0.1)
    school_count = st.number_input("Количество школ в радиусе", 0, 20, 3)
    school_dist = st.number_input("Расстояние до ближайшей (mi)", 0.0, 50.0, 1.0, 0.1)
    school_hs = st.checkbox("Есть старшая школа поблизости", value=True)
    
    st.subheader("Цена объявления")
    ask_price = st.number_input(
        "Цена в объявлении ($)",
        min_value=0.0,
        value=0.0,
        step=10000.0,
        help="Введите цену из объявления, чтобы сравнить с оценкой модели.")

# Кнопка и вывод результата

st.divider()
run_btn = st.button("Оценить стоимость", type="primary", use_container_width=True)

if run_btn:
    # Проверка обязательных полей геолокации
    if not state or not city or not zipcode:
        st.error("Пожалуйста, выберите штат, город и ZIP-код.")
    else:
        with st.spinner("Расчёт оценочной стоимости..."):
            # Собираем словарь с сырыми признаками
            raw_data = {
                "sqft": sqft,
                "beds": beds,
                "baths": baths,
                "stories": stories,
                "propertyType": prop_type,
                "state": state,
                "city": city,
                "zipcode": str(zipcode),
                "hfyearbuilt": year_built if year_built > 0 else np.nan,
                "hfremodeledyear": np.nan,  # в упрощённой форме не вводим
                "hflotsqft": lot_size if lot_size > 0 else np.nan,
                "hfparkingspaces": parking_spaces if parking_spaces > 0 else np.nan,
                "hfhasgarage": int(has_garage),
                "hfhascarport": int(has_carport),
                "hfhasstreet": 0,
                "hfhaspool": int(has_pool),
                "hfhasfireplace": int(has_fireplace),
                "hfhascentralac": int(has_ac),
                "hfhasforcedair": int(has_fa),
                "schoolratingavg": school_avg if school_avg > 0 else np.nan,
                "schoolratingmax": school_max if school_max > 0 else np.nan,
                "schoolratingcount": float(school_count),
                "schoolmindistmi": school_dist if school_dist > 0 else np.nan,
                "schoolhashighschool": int(school_hs)}

            X_input = build_inference_frame(raw_data, bundle)
            pred_log = bundle["model"].predict(X_input)[0]
            pred_price = float(np.expm1(np.clip(pred_log, 0, 20)))

        # Вывод результатов
        st.success(f"Оценочная рыночная стоимость: **${pred_price:,.0f}**")

        if ask_price > 0:
            delta = ask_price - pred_price
            delta_pct = (delta / pred_price) * 100 if pred_price > 0 else 0.0

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Цена в объявлении", f"${ask_price:,.0f}")
            with col2:
                st.metric("Отклонение", f"${delta:+,.0f}", f"{delta_pct:+.1f}%")
            with col3:
                if delta > 0:
                    st.warning("Переоценён")
                elif delta < 0:
                    st.success("Недооценён")
                else:
                    st.info("Совпадает")

            if delta > 0:
                st.warning("Объект, возможно, **переоценён**. Стоит проверить адекватность цены.")
            elif delta < 0:
                st.success("Объект выглядит **недооценённым**. Потенциально выгодное предложение!")
            else:
                st.info("Цена совпадает с оценкой модели.")
        else:
            st.info("Укажите цену объявления, чтобы увидеть, насколько предложение выгодно.")

        # Дополнительная информация
        with st.expander("Детали прогноза"):
            st.write(f"Модель: **{bundle['model_name']}**")
            st.write(f"Reference year: {bundle['reference_year']}")
            st.write(f"Использовано признаков: {len(bundle['feature_columns'])}")