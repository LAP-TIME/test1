"""
모델링 모듈

칼로리 예측을 위한 회귀 모델 학습 및 평가
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')

# XGBoost는 선택적으로 import
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class CaloriePredictor:
    """칼로리 예측 모델 클래스"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        self.models = self._initialize_models()

    def _initialize_models(self) -> Dict[str, Any]:
        """사용할 모델들 초기화"""
        models = {
            'Ridge': Ridge(random_state=self.random_state),
            'Lasso': Lasso(random_state=self.random_state),
            'ElasticNet': ElasticNet(random_state=self.random_state),
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100,
                random_state=self.random_state
            ),
        }

        if XGBOOST_AVAILABLE:
            models['XGBoost'] = XGBRegressor(
                n_estimators=100,
                random_state=self.random_state,
                verbosity=0
            )

        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = LGBMRegressor(
                n_estimators=100,
                random_state=self.random_state,
                verbose=-1
            )

        return models

    def prepare_data(self, df: pd.DataFrame,
                     target_col: str = 'calories',
                     test_size: float = 0.2,
                     drop_cols: Optional[List[str]] = None) -> Tuple:
        """
        데이터 분할 및 스케일링

        Args:
            df: 전처리된 DataFrame
            target_col: 타겟 변수명
            test_size: 테스트 데이터 비율
            drop_cols: 제외할 컬럼 리스트

        Returns:
            X_train, X_test, y_train, y_test (스케일링된 데이터)
        """
        df = df.copy()

        # 제외할 컬럼 처리
        if drop_cols is None:
            drop_cols = ['user_id']
        drop_cols = [col for col in drop_cols if col in df.columns]

        # 특성과 타겟 분리
        X = df.drop([target_col] + drop_cols, axis=1)
        y = df[target_col]

        self.feature_names = X.columns.tolist()

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        # 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print("=" * 60)
        print("📊 데이터 준비 완료")
        print("=" * 60)
        print(f"  학습 데이터: {X_train_scaled.shape[0]} samples")
        print(f"  테스트 데이터: {X_test_scaled.shape[0]} samples")
        print(f"  특성 수: {X_train_scaled.shape[1]}")
        print(f"  특성 목록: {self.feature_names}")

        return X_train_scaled, X_test_scaled, y_train, y_test

    def evaluate_model(self, y_true: np.ndarray,
                       y_pred: np.ndarray) -> Dict[str, float]:
        """
        모델 평가 지표 계산

        평가 지표:
        - MAE: 평균 절대 오차 (해석 용이)
        - RMSE: 평균 제곱근 오차 (큰 오차에 민감)
        - R²: 결정계수 (설명력)
        - MAPE: 평균 절대 백분율 오차 (상대적 오차)
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        # MAPE 계산 (0으로 나누는 것 방지)
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }

    def compare_models(self, X_train: np.ndarray, X_test: np.ndarray,
                       y_train: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        여러 모델 비교 평가
        """
        print("\n" + "=" * 60)
        print("🔬 모델 비교 평가")
        print("=" * 60)

        results = []

        for name, model in self.models.items():
            print(f"\n  Training {name}...", end=" ")

            # 학습
            model.fit(X_train, y_train)

            # 예측
            y_pred = model.predict(X_test)

            # 평가
            metrics = self.evaluate_model(y_test, y_pred)

            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                        scoring='neg_mean_absolute_error')
            cv_mae = -cv_scores.mean()

            results.append({
                'Model': name,
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE'],
                'R2': metrics['R2'],
                'MAPE': metrics['MAPE'],
                'CV_MAE': cv_mae
            })

            print(f"Done! R² = {metrics['R2']:.4f}")

        # 결과 DataFrame
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('MAE').reset_index(drop=True)

        # 최고 모델 저장
        best_idx = results_df['MAE'].idxmin()
        self.best_model_name = results_df.loc[best_idx, 'Model']
        self.best_model = self.models[self.best_model_name]

        print("\n" + "=" * 60)
        print("📊 모델 비교 결과 (MAE 기준 정렬)")
        print("=" * 60)
        print(results_df.to_string(index=False))
        print(f"\n🏆 최고 모델: {self.best_model_name}")

        return results_df

    def tune_hyperparameters(self, X_train: np.ndarray,
                             y_train: np.ndarray,
                             model_name: str = 'RandomForest') -> Any:
        """
        그리드 서치를 통한 하이퍼파라미터 튜닝
        """
        print("\n" + "=" * 60)
        print(f"🔧 하이퍼파라미터 튜닝: {model_name}")
        print("=" * 60)

        param_grids = {
            'RandomForest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'GradientBoosting': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'min_samples_split': [2, 5]
            },
            'XGBoost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0]
            },
            'LightGBM': {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, -1],
                'learning_rate': [0.01, 0.1],
                'num_leaves': [31, 50]
            }
        }

        if model_name not in param_grids:
            print(f"  ⚠️ {model_name}에 대한 파라미터 그리드가 정의되지 않음")
            return self.models[model_name]

        grid_search = GridSearchCV(
            self.models[model_name],
            param_grids[model_name],
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        print(f"\n  최적 파라미터: {grid_search.best_params_}")
        print(f"  최적 CV MAE: {-grid_search.best_score_:.2f} kcal")

        self.best_model = grid_search.best_estimator_
        self.best_model_name = model_name

        return grid_search.best_estimator_

    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        특성 중요도 분석

        어떤 변수가 칼로리 예측에 가장 중요한지 확인
        """
        if self.best_model is None:
            print("  ⚠️ 먼저 모델을 학습시켜야 합니다.")
            return None

        print("\n" + "=" * 60)
        print("📊 특성 중요도 분석")
        print("=" * 60)

        # Tree 기반 모델만 feature_importances_ 지원
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            importance = np.abs(self.best_model.coef_)
        else:
            print("  ⚠️ 이 모델은 특성 중요도를 지원하지 않습니다.")
            return None

        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)

        print(f"\n  Top {top_n} 중요 특성:")
        for idx, row in importance_df.head(top_n).iterrows():
            bar = '█' * int(row['Importance'] * 50)
            print(f"  {row['Feature']:30s} {row['Importance']:.4f} {bar}")

        return importance_df

    def save_model(self, filepath: str = 'models/calorie_model.pkl') -> None:
        """모델 저장"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }

        joblib.dump(model_data, filepath)
        print(f"\n💾 모델 저장됨: {filepath}")

    def load_model(self, filepath: str = 'models/calorie_model.pkl') -> None:
        """모델 로드"""
        model_data = joblib.load(filepath)

        self.best_model = model_data['model']
        self.best_model_name = model_data['model_name']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']

        print(f"📂 모델 로드됨: {filepath}")
        print(f"  모델: {self.best_model_name}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """새로운 데이터에 대한 예측"""
        if self.best_model is None:
            raise ValueError("모델이 학습되지 않았습니다.")

        # 특성 순서 맞추기
        X = X[self.feature_names]
        X_scaled = self.scaler.transform(X)

        return self.best_model.predict(X_scaled)


def train_and_evaluate(df: pd.DataFrame,
                       target_col: str = 'calories',
                       tune: bool = False) -> CaloriePredictor:
    """
    전체 학습 파이프라인 실행

    Args:
        df: 전처리 및 특성 공학이 완료된 DataFrame
        target_col: 타겟 변수명
        tune: 하이퍼파라미터 튜닝 여부

    Returns:
        학습된 CaloriePredictor 객체
    """
    predictor = CaloriePredictor()

    # 데이터 준비
    X_train, X_test, y_train, y_test = predictor.prepare_data(df, target_col)

    # 모델 비교
    results = predictor.compare_models(X_train, X_test, y_train, y_test)

    # 하이퍼파라미터 튜닝 (선택적)
    if tune:
        predictor.tune_hyperparameters(X_train, y_train, predictor.best_model_name)

        # 튜닝 후 재평가
        y_pred = predictor.best_model.predict(X_test)
        final_metrics = predictor.evaluate_model(y_test, y_pred)

        print("\n" + "=" * 60)
        print("📊 튜닝 후 최종 성능")
        print("=" * 60)
        for metric, value in final_metrics.items():
            print(f"  {metric}: {value:.4f}")

    # 특성 중요도
    predictor.get_feature_importance()

    # 모델 저장
    predictor.save_model()

    return predictor


if __name__ == "__main__":
    from data_preprocessing import load_and_preprocess
    from feature_engineering import engineer_features

    # 데이터 로드 및 전처리
    df = load_and_preprocess("data/raw/calories_burned_data.csv")
    df = engineer_features(df)

    # 모델 학습 및 평가
    predictor = train_and_evaluate(df, tune=False)
