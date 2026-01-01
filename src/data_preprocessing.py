"""
데이터 전처리 모듈
- 단위 변환 (미국식 → 국제 단위)
- 결측치 처리
- 이상치 탐지 및 처리
- 범주형 변수 인코딩
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


class DataPreprocessor:
    """칼로리 데이터 전처리 클래스"""

    def __init__(self):
        self.weight_status_mapping = {
            'Underweight': 0,
            'Normal': 1,
            'Overweight': 2,
            'Obese': 3
        }
        self.gender_mapping = {'F': 0, 'M': 1}

    # ================================================================
    # STEP 1: 단위 변환 (매우 중요!)
    # ================================================================

    def convert_height_to_cm(self, feet: float, inches: float) -> float:
        """
        키 변환: 피트 + 인치 → cm

        공식: 총 인치 = (피트 × 12) + 인치
              cm = 총 인치 × 2.54

        예시: 5피트 10인치 = (5×12 + 10) × 2.54 = 177.8 cm
        """
        total_inches = (feet * 12) + inches
        return total_inches * 2.54

    def convert_weight_to_kg(self, weight_lb: float) -> float:
        """
        몸무게 변환: 파운드(lb) → kg

        공식: kg = lb × 0.453592

        예시: 176 lb = 176 × 0.453592 = 79.83 kg
        """
        return weight_lb * 0.453592

    def convert_temp_to_celsius(self, temp_f: float) -> float:
        """
        체온 변환: 화씨(°F) → 섭씨(°C)

        공식: °C = (°F - 32) × 5/9

        예시: 101.2°F = (101.2 - 32) × 5/9 = 38.44°C
        """
        return (temp_f - 32) * 5 / 9

    # ================================================================
    # STEP 2: 결측치 처리
    # ================================================================

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        결측치 처리 전략:
        - 수치형: 중앙값(median) 대체 (이상치에 강건함)
        - 범주형: 최빈값(mode) 대체
        """
        df = df.copy()

        # 결측치 현황 출력
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print("📊 결측치 현황:")
            print(missing[missing > 0])

        # 수치형 변수: 중앙값 대체
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"  - {col}: 중앙값 {median_val:.2f}로 대체")

        # 범주형 변수: 최빈값 대체
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                print(f"  - {col}: 최빈값 '{mode_val}'로 대체")

        return df

    # ================================================================
    # STEP 3: 이상치 탐지 및 처리
    # ================================================================

    def detect_outliers_iqr(self, df: pd.DataFrame, column: str) -> pd.Series:
        """
        IQR 방식으로 이상치 탐지

        이상치 기준: Q1 - 1.5×IQR 미만 또는 Q3 + 1.5×IQR 초과
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        return outliers

    def handle_outliers(self, df: pd.DataFrame,
                        columns: list,
                        method: str = 'clip') -> pd.DataFrame:
        """
        이상치 처리

        Args:
            method: 'clip' (경계값으로 대체) 또는 'remove' (제거)
        """
        df = df.copy()

        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

            if outlier_count > 0:
                if method == 'clip':
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    print(f"  - {col}: {outlier_count}개 이상치를 경계값으로 조정")
                elif method == 'remove':
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                    print(f"  - {col}: {outlier_count}개 이상치 행 제거")

        return df

    # ================================================================
    # STEP 4: 범주형 변수 인코딩
    # ================================================================

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        범주형 변수 인코딩

        - gender: M/F → 1/0 (Label Encoding)
        - weight_status: Ordinal Encoding (순서가 있는 범주)
        """
        df = df.copy()

        # 성별 인코딩
        if 'gender' in df.columns:
            df['gender_encoded'] = df['gender'].map(self.gender_mapping)
            print(f"  - gender: {self.gender_mapping}")

        # 체중 상태 인코딩 (순서형)
        if 'weight_status' in df.columns:
            df['weight_status_encoded'] = df['weight_status'].map(self.weight_status_mapping)
            print(f"  - weight_status: {self.weight_status_mapping}")

        return df

    # ================================================================
    # 통합 전처리 파이프라인
    # ================================================================

    def preprocess(self, df: pd.DataFrame,
                   handle_outliers_method: str = 'clip') -> pd.DataFrame:
        """
        전체 전처리 파이프라인 실행
        """
        print("=" * 60)
        print("🔧 데이터 전처리 시작")
        print("=" * 60)

        df = df.copy()
        original_shape = df.shape

        # 1. 단위 변환
        print("\n📐 STEP 1: 단위 변환")

        # 키 변환 (피트 + 인치 → cm)
        if 'height_feet' in df.columns and 'height_inches' in df.columns:
            df['height_cm'] = df.apply(
                lambda x: self.convert_height_to_cm(x['height_feet'], x['height_inches']),
                axis=1
            )
            print(f"  - 키: 피트+인치 → cm 변환 완료")
            # 원본 컬럼 제거 (다중공선성 방지)
            df.drop(['height_feet', 'height_inches'], axis=1, inplace=True)

        # 몸무게 변환 (파운드 → kg)
        if 'weight_lb' in df.columns:
            df['weight_kg'] = df['weight_lb'].apply(self.convert_weight_to_kg)
            print(f"  - 몸무게: 파운드 → kg 변환 완료")
            df.drop('weight_lb', axis=1, inplace=True)

        # 체온 변환 (화씨 → 섭씨)
        if 'body_temp_f' in df.columns:
            df['body_temp_c'] = df['body_temp_f'].apply(self.convert_temp_to_celsius)
            print(f"  - 체온: 화씨 → 섭씨 변환 완료")
            df.drop('body_temp_f', axis=1, inplace=True)

        # 2. 결측치 처리
        print("\n🔍 STEP 2: 결측치 처리")
        df = self.handle_missing_values(df)

        # 3. 이상치 처리
        print("\n📊 STEP 3: 이상치 처리")
        numeric_cols = ['age', 'duration', 'heart_rate', 'calories',
                        'height_cm', 'weight_kg', 'body_temp_c']
        numeric_cols = [col for col in numeric_cols if col in df.columns]
        df = self.handle_outliers(df, numeric_cols, method=handle_outliers_method)

        # 4. 범주형 변수 인코딩
        print("\n🏷️ STEP 4: 범주형 변수 인코딩")
        df = self.encode_categorical(df)

        # 원본 범주형 컬럼 제거
        cols_to_drop = ['gender', 'weight_status']
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        df.drop(cols_to_drop, axis=1, inplace=True)

        print("\n" + "=" * 60)
        print(f"✅ 전처리 완료: {original_shape} → {df.shape}")
        print("=" * 60)

        return df


def load_and_preprocess(filepath: str) -> pd.DataFrame:
    """데이터 로드 및 전처리 편의 함수"""
    df = pd.read_csv(filepath)
    preprocessor = DataPreprocessor()
    return preprocessor.preprocess(df)


if __name__ == "__main__":
    # 테스트
    df = pd.read_csv("data/raw/calories_burned_data.csv")
    print(f"원본 데이터 shape: {df.shape}")
    print(f"원본 컬럼: {df.columns.tolist()}")

    preprocessor = DataPreprocessor()
    df_processed = preprocessor.preprocess(df)

    print(f"\n처리된 데이터 컬럼: {df_processed.columns.tolist()}")
    print(df_processed.head())
