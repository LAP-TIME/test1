#!/usr/bin/env python3
"""
칼로리 소모량 예측 모델 - 메인 실행 파일

실행 방법:
    python main.py                    # 전체 파이프라인 실행
    python main.py --eda-only         # EDA만 실행
    python main.py --tune             # 하이퍼파라미터 튜닝 포함
    python main.py --predict          # 저장된 모델로 예측
"""

import argparse
import sys
import os

# src 폴더를 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor, load_and_preprocess
from feature_engineering import FeatureEngineer, engineer_features
from eda import ExploratoryDataAnalysis
from model import CaloriePredictor, train_and_evaluate

import pandas as pd


def run_full_pipeline(data_path: str, tune: bool = False) -> None:
    """
    전체 ML 파이프라인 실행

    1. 데이터 로드
    2. 전처리 (단위 변환, 결측치, 이상치, 인코딩)
    3. 특성 공학
    4. EDA
    5. 모델 학습 및 평가
    6. 모델 저장
    """
    print("\n" + "=" * 70)
    print("🔥 칼로리 소모량 예측 모델 - 전체 파이프라인")
    print("=" * 70)

    # Step 1: 데이터 로드 및 전처리
    print("\n📌 STEP 1: 데이터 전처리")
    df = load_and_preprocess(data_path)

    # Step 2: 특성 공학
    print("\n📌 STEP 2: 특성 공학")
    df = engineer_features(df)

    # Step 3: 전처리된 데이터 저장
    processed_path = 'data/processed/calories_processed.csv'
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"\n💾 전처리된 데이터 저장됨: {processed_path}")

    # Step 4: EDA
    print("\n📌 STEP 3: 탐색적 데이터 분석 (EDA)")
    eda = ExploratoryDataAnalysis()
    eda.run_full_eda(df, target_col='calories', output_dir='outputs/eda')

    # Step 5: 모델 학습 및 평가
    print("\n📌 STEP 4: 모델 학습 및 평가")
    predictor = train_and_evaluate(df, target_col='calories', tune=tune)

    print("\n" + "=" * 70)
    print("✅ 파이프라인 완료!")
    print("=" * 70)
    print(f"\n📂 생성된 파일:")
    print(f"   - 전처리 데이터: {processed_path}")
    print(f"   - EDA 결과: outputs/eda/")
    print(f"   - 학습된 모델: models/calorie_model.pkl")


def run_eda_only(data_path: str) -> None:
    """EDA만 실행"""
    print("\n" + "=" * 70)
    print("📊 탐색적 데이터 분석 (EDA)")
    print("=" * 70)

    df = load_and_preprocess(data_path)
    df = engineer_features(df)

    eda = ExploratoryDataAnalysis()
    eda.run_full_eda(df, target_col='calories', output_dir='outputs/eda')


def run_prediction_demo() -> None:
    """저장된 모델을 사용한 예측 데모"""
    print("\n" + "=" * 70)
    print("🔮 칼로리 예측 데모")
    print("=" * 70)

    # 모델 로드
    predictor = CaloriePredictor()
    try:
        predictor.load_model('models/calorie_model.pkl')
    except FileNotFoundError:
        print("⚠️ 저장된 모델이 없습니다. 먼저 학습을 실행하세요.")
        return

    # 샘플 데이터 (전처리 + 특성공학 완료된 형태)
    # 실제 사용시에는 새로운 데이터를 전처리 파이프라인에 통과시켜야 함
    sample_data = pd.DataFrame([{
        'age': 30,
        'duration': 45,
        'heart_rate': 130,
        'height_cm': 175.0,
        'weight_kg': 75.0,
        'body_temp_c': 39.0,
        'gender_encoded': 1,
        'weight_status_encoded': 1,
        'bmi': 24.5,
        'max_heart_rate': 187,
        'hr_intensity': 69.5,
        'hr_zone': 2,
        'exercise_intensity': 45.0,
        'bmr': 1750.0,
        'bsa': 1.9,
        'duration_hr_interaction': 5850,
        'weight_duration_interaction': 3375.0,
        'age_group_middle': False,
        'age_group_senior': False,
        'age_group_young': True
    }])

    # 필요한 특성만 선택
    sample_data = sample_data[predictor.feature_names]

    # 예측
    predicted_calories = predictor.predict(sample_data)[0]

    print(f"\n📋 입력 데이터:")
    print(f"   - 나이: 30세")
    print(f"   - 성별: 남성")
    print(f"   - 키: 175cm")
    print(f"   - 체중: 75kg")
    print(f"   - 운동 시간: 45분")
    print(f"   - 평균 심박수: 130 BPM")
    print(f"   - 운동 중 체온: 39°C")

    print(f"\n🔥 예측 소모 칼로리: {predicted_calories:.1f} kcal")


def main():
    parser = argparse.ArgumentParser(
        description='칼로리 소모량 예측 모델',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python main.py                           # 전체 파이프라인
  python main.py --data path/to/data.csv   # 커스텀 데이터
  python main.py --eda-only                # EDA만 실행
  python main.py --tune                    # 하이퍼파라미터 튜닝 포함
  python main.py --predict                 # 예측 데모
        """
    )

    parser.add_argument(
        '--data', '-d',
        type=str,
        default='data/raw/calories_burned_data.csv',
        help='데이터 파일 경로'
    )
    parser.add_argument(
        '--eda-only',
        action='store_true',
        help='EDA만 실행'
    )
    parser.add_argument(
        '--tune',
        action='store_true',
        help='하이퍼파라미터 튜닝 수행'
    )
    parser.add_argument(
        '--predict',
        action='store_true',
        help='저장된 모델로 예측 데모 실행'
    )

    args = parser.parse_args()

    if args.predict:
        run_prediction_demo()
    elif args.eda_only:
        run_eda_only(args.data)
    else:
        run_full_pipeline(args.data, tune=args.tune)


if __name__ == '__main__':
    main()
