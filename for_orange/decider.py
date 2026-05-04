import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class StripPredictor:
    """Предсказание центра strip для множества strip на одном изображении с фильтрацией"""

    def __init__(self, yolo_threshold=0.6, cls_threshold=0.5, min_confidence=0.7):
        self.yolo_threshold = yolo_threshold
        self.cls_threshold = cls_threshold
        self.min_confidence = min_confidence  # Минимальная уверенность для принятия strip
        self.clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _extract_features(self, df):
        """Извлечение признаков для классификатора"""
        df = df.copy()
        df['yolo_center'] = df['yolo_strip_y'] + df['yolo_strip_h'] / 2
        df['center_diff'] = np.abs(df['yolo_center'] - df['cls_strip_center_y'])
        df['yolo_size'] = df['yolo_strip_w'] * df['yolo_strip_h']

        features = df[['yolo_strip_conf', 'cls_strip_center_y_score', 'center_diff', 'yolo_size']].fillna(0)
        return features.values

    def fit(self, df):
        """Обучение на данных с ann_strip_y"""
        train = df.dropna(subset=['ann_strip_y']).copy()

        if len(train) < 10:
            print("Недостаточно данных для обучения, используются только правила")
            return self

        train['yolo_center'] = train['yolo_strip_y'] + train['yolo_strip_h'] / 2
        yolo_err = np.abs(train['yolo_center'] - train['ann_strip_y'])
        cls_err = np.abs(train['cls_strip_center_y'] - train['ann_strip_y'])
        y_target = (yolo_err < cls_err).astype(int)

        X = self._extract_features(train)
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y_target)
        self.is_fitted = True
        return self

    def predict(self, df):
        """
        Предсказание для множества strip на одном или нескольких изображениях
        Возвращает: numpy array с предсказаниями для каждой строки (NaN для отсеянных)
        """
        yolo_center = df['yolo_strip_y'] + df['yolo_strip_h'] / 2
        cls_center = df['cls_strip_center_y']

        predictions = []
        for i in range(len(df)):
            has_yolo = pd.notna(df['yolo_strip_conf'].iloc[i])
            has_cls = pd.notna(df['cls_strip_center_y_score'].iloc[i])

            # Фильтрация по минимальной уверенности
            yolo_conf = df['yolo_strip_conf'].iloc[i] if has_yolo else 0
            cls_score = df['cls_strip_center_y_score'].iloc[i] if has_cls else 0

            # Если оба источника имеют низкую уверенность - пропускаем
            if has_yolo and yolo_conf < self.min_confidence and has_cls and cls_score < self.min_confidence:
                predictions.append(np.nan)
                continue

            # Если только YOLO, но уверенность太低
            if has_yolo and not has_cls and yolo_conf < self.min_confidence:
                predictions.append(np.nan)
                continue

            # Если только CLS, но уверенность太低
            if not has_yolo and has_cls and cls_score < self.min_confidence:
                predictions.append(np.nan)
                continue

            # Выбор лучшего предсказания
            if has_yolo and not has_cls:
                predictions.append(yolo_center.iloc[i])
            elif not has_yolo and has_cls:
                predictions.append(cls_center.iloc[i])
            elif has_yolo and has_cls:
                if self.is_fitted:
                    X = self._extract_features(df.iloc[[i]])
                    X_scaled = self.scaler.transform(X)
                    use_yolo = self.clf.predict(X_scaled)[0]
                    predictions.append(yolo_center.iloc[i] if use_yolo else cls_center.iloc[i])
                else:
                    if yolo_conf > self.yolo_threshold:
                        predictions.append(yolo_center.iloc[i])
                    elif cls_score > self.cls_threshold:
                        predictions.append(cls_center.iloc[i])
                    else:
                        predictions.append((yolo_center.iloc[i] + cls_center.iloc[i]) / 2)
            else:
                predictions.append(np.nan)

        return np.array(predictions)

    def predict_filtered(self, df):
        """
        Предсказание с возвратом только отфильтрованных (не NaN) результатов
        Возвращает: tuple (filtered_predictions, filtered_indices)
        """
        predictions = self.predict(df)
        valid_mask = ~np.isnan(predictions)
        return predictions[valid_mask], np.where(valid_mask)[0]


