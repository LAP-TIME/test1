"""
특성 공학 (Feature Engineering) 모듈

도메인 지식을 활용하여 칼로리 예측에 유용한 파생 변수 생성
"""

import pandas as pd
import numpy as np
from typing import Optional


class FeatureEngineer:
    """칼로리 예측을 위한 특성 공학 클래스"""

    def __init__(self):
        pass

    # ================================================================
    # 도메인 지식 기반 파생 변수 생성
    # ================================================================

    def calculate_bmi(self, weight_kg: float, height_cm: float) -> float:
        """
        BMI (Body Mass Index) 계산

        공식: BMI = 체중(kg) / 키(m)²

        해석:
        - < 18.5: 저체중
        - 18.5 ~ 24.9: 정상
        - 25.0 ~ 29.9: 과체중
        - ≥ 30: 비만

        칼로리 소모 관련성: BMI가 높을수록 같은 운동에 더 많은 칼로리 소모
        """
        height_m = height_cm / 100
        return weight_kg / (height_m ** 2)

    def calculate_max_heart_rate(self, age: int) -> int:
        """
        최대 심박수 추정 (Tanaka 공식)

        공식: 최대 심박수 = 208 - (0.7 × 나이)

        참고: 기존 공식 (220 - 나이)보다 정확하다고 알려짐
        """
        return int(208 - (0.7 * age))

    def calculate_heart_rate_intensity(self, heart_rate: float,
                                       max_heart_rate: float) -> float:
        """
        심박수 강도 (운동 강도 지표)

        공식: 강도 = 현재 심박수 / 최대 심박수 × 100

        해석:
        - 50-60%: 저강도 (지방 연소)
        - 60-70%: 중저강도 (유산소)
        - 70-80%: 중고강도 (심폐 지구력)
        - 80-90%: 고강도 (무산소 역치)
        - 90%+: 최대 강도

        칼로리 소모 관련성: 강도가 높을수록 분당 칼로리 소모 증가
        """
        return (heart_rate / max_heart_rate) * 100

    def calculate_calories_per_minute(self, calories: float,
                                      duration: float) -> float:
        """
        분당 칼로리 소모량

        칼로리 소모 관련성: 운동 효율성 지표
        """
        if duration == 0:
            return 0
        return calories / duration

    def calculate_exercise_intensity_score(self, heart_rate: float,
                                           body_temp_c: float,
                                           duration: float) -> float:
        """
        운동 강도 종합 점수

        심박수, 체온, 운동 시간을 종합한 운동 강도 지표
        - 심박수 높을수록 ↑
        - 체온 높을수록 ↑ (대사율 증가 반영)
        - 운동 시간 길수록 ↑ (누적 효과)
        """
        # 정규화 (대략적인 범위 기준)
        hr_normalized = heart_rate / 200  # 심박수 0~200 가정
        temp_normalized = (body_temp_c - 36) / 5  # 체온 36~41도 가정
        duration_normalized = duration / 60  # 60분 기준

        return (hr_normalized * 0.5 + temp_normalized * 0.3 + duration_normalized * 0.2) * 100

    def calculate_bmr(self, weight_kg: float, height_cm: float,
                      age: int, gender: int) -> float:
        """
        기초 대사량 (BMR) 계산 - Mifflin-St Jeor 공식

        남성: BMR = (10 × 체중kg) + (6.25 × 키cm) - (5 × 나이) + 5
        여성: BMR = (10 × 체중kg) + (6.25 × 키cm) - (5 × 나이) - 161

        칼로리 소모 관련성: 기초대사량이 높은 사람은 운동 중에도 더 많은 칼로리 소모
        """
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age)

        if gender == 1:  # Male
            bmr += 5
        else:  # Female
            bmr -= 161

        return bmr

    def calculate_body_surface_area(self, weight_kg: float,
                                    height_cm: float) -> float:
        """
        체표면적 (BSA) 계산 - Du Bois 공식

        공식: BSA = 0.007184 × 체중^0.425 × 키^0.725

        칼로리 소모 관련성: 체표면적이 클수록 열 발산이 많아 칼로리 소모 증가
        """
        return 0.007184 * (weight_kg ** 0.425) * (height_cm ** 0.725)

    def get_age_group(self, age: int) -> str:
        """나이 그룹화"""
        if age < 30:
            return 'young'
        elif age < 50:
            return 'middle'
        else:
            return 'senior'

    def get_hr_zone(self, hr_intensity: float) -> int:
        """
        심박수 구간 (Heart Rate Zone)

        1: 50-60% (회복)
        2: 60-70% (지방 연소)
        3: 70-80% (유산소)
        4: 80-90% (무산소 역치)
        5: 90%+ (최대)
        """
        if hr_intensity < 60:
            return 1
        elif hr_intensity < 70:
            return 2
        elif hr_intensity < 80:
            return 3
        elif hr_intensity < 90:
            return 4
        else:
            return 5

    # ================================================================
    # 통합 특성 생성 파이프라인
    # ================================================================

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        모든 파생 변수 생성
        """
        print("=" * 60)
        print("🔬 특성 공학 (Feature Engineering) 시작")
        print("=" * 60)

        df = df.copy()
        original_features = len(df.columns)

        # 1. BMI 계산
        if 'weight_kg' in df.columns and 'height_cm' in df.columns:
            df['bmi'] = df.apply(
                lambda x: self.calculate_bmi(x['weight_kg'], x['height_cm']),
                axis=1
            )
            print("  ✓ BMI 생성")

        # 2. 최대 심박수 계산
        if 'age' in df.columns:
            df['max_heart_rate'] = df['age'].apply(self.calculate_max_heart_rate)
            print("  ✓ 최대 심박수 생성")

        # 3. 심박수 강도 계산
        if 'heart_rate' in df.columns and 'max_heart_rate' in df.columns:
            df['hr_intensity'] = df.apply(
                lambda x: self.calculate_heart_rate_intensity(
                    x['heart_rate'], x['max_heart_rate']
                ),
                axis=1
            )
            print("  ✓ 심박수 강도(%) 생성")

            # 심박수 구간
            df['hr_zone'] = df['hr_intensity'].apply(self.get_hr_zone)
            print("  ✓ 심박수 구간(Zone) 생성")

        # 4. 운동 강도 종합 점수
        if all(col in df.columns for col in ['heart_rate', 'body_temp_c', 'duration']):
            df['exercise_intensity'] = df.apply(
                lambda x: self.calculate_exercise_intensity_score(
                    x['heart_rate'], x['body_temp_c'], x['duration']
                ),
                axis=1
            )
            print("  ✓ 운동 강도 종합 점수 생성")

        # 5. BMR (기초대사량) 계산
        if all(col in df.columns for col in ['weight_kg', 'height_cm', 'age', 'gender_encoded']):
            df['bmr'] = df.apply(
                lambda x: self.calculate_bmr(
                    x['weight_kg'], x['height_cm'], x['age'], x['gender_encoded']
                ),
                axis=1
            )
            print("  ✓ 기초대사량(BMR) 생성")

        # 6. 체표면적 계산
        if 'weight_kg' in df.columns and 'height_cm' in df.columns:
            df['bsa'] = df.apply(
                lambda x: self.calculate_body_surface_area(x['weight_kg'], x['height_cm']),
                axis=1
            )
            print("  ✓ 체표면적(BSA) 생성")

        # 7. 상호작용 특성 (Interaction Features)
        if 'duration' in df.columns and 'heart_rate' in df.columns:
            df['duration_hr_interaction'] = df['duration'] * df['heart_rate']
            print("  ✓ 운동시간×심박수 상호작용 생성")

        if 'weight_kg' in df.columns and 'duration' in df.columns:
            df['weight_duration_interaction'] = df['weight_kg'] * df['duration']
            print("  ✓ 체중×운동시간 상호작용 생성")

        # 8. 나이 그룹
        if 'age' in df.columns:
            df['age_group'] = df['age'].apply(self.get_age_group)
            # One-hot encoding
            age_dummies = pd.get_dummies(df['age_group'], prefix='age_group')
            df = pd.concat([df, age_dummies], axis=1)
            df.drop('age_group', axis=1, inplace=True)
            print("  ✓ 나이 그룹 (One-Hot) 생성")

        new_features = len(df.columns) - original_features
        print("\n" + "=" * 60)
        print(f"✅ 특성 공학 완료: {new_features}개 새로운 특성 생성")
        print(f"   총 특성 수: {len(df.columns)}개")
        print("=" * 60)

        return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """특성 공학 편의 함수"""
    engineer = FeatureEngineer()
    return engineer.create_features(df)


if __name__ == "__main__":
    # 테스트
    from data_preprocessing import load_and_preprocess

    df = load_and_preprocess("data/raw/calories_burned_data.csv")
    df_features = engineer_features(df)

    print(f"\n최종 컬럼: {df_features.columns.tolist()}")
    print(df_features.head())
