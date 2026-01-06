"""
칼로리 소모량 예측 모델 - 개선 버전 (v2)
==========================================
기존 RMSE: 0.10964 (공동 1등)
목표: 버려진 피처(BMI, Body_Temperature, Weight_Status) 추가로 성능 개선

변경사항:
- 피처 5개 → 9개
- BMI 계산 추가
- Height_Total_Inches 계산 추가
- Weight_Status 인코딩 추가
"""

# =============================================================================
# 1. 라이브러리 임포트
# =============================================================================
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("칼로리 소모량 예측 모델 - 개선 버전 (v2)")
print("=" * 60)

# =============================================================================
# 2. Google Drive 마운트
# =============================================================================
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("\n✅ Google Drive 마운트 완료")
except:
    print("\n⚠️ Google Colab 환경이 아닙니다. 로컬 경로를 사용합니다.")

# =============================================================================
# 3. 데이터 로드
# =============================================================================
print("\n" + "=" * 60)
print("=== 데이터 로드 ===")
print("=" * 60)

# 데이터 경로 설정 (Google Drive)
DATA_PATH = '/content/drive/MyDrive/AI_Projects/calorie_prediction_model/data/'
OUTPUT_PATH = '/content/drive/MyDrive/AI_Projects/calorie_prediction_model/'

try:
    train = pd.read_csv(DATA_PATH + 'train.csv')
    test = pd.read_csv(DATA_PATH + 'test.csv')
    submit = pd.read_csv(DATA_PATH + 'sample_submission.csv')

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Sample submission shape: {submit.shape}")
    print("\n✅ 데이터 로드 완료")

except FileNotFoundError as e:
    print(f"❌ 파일을 찾을 수 없습니다: {e}")
    print("경로를 확인해주세요.")
    raise

# =============================================================================
# 4. 간단한 EDA
# =============================================================================
print("\n" + "=" * 60)
print("=== 데이터 미리보기 ===")
print("=" * 60)

print("\n[Train 데이터 상위 5개 행]")
print(train.head())

print("\n[컬럼 목록]")
print(f"Train 컬럼: {list(train.columns)}")
print(f"Test 컬럼: {list(test.columns)}")

print("\n[결측치 확인]")
print(f"Train 결측치: {train.isnull().sum().sum()}")
print(f"Test 결측치: {test.isnull().sum().sum()}")

print("\n[목표 변수 통계]")
print(train['Calories_Burned'].describe())

print("\n[Weight_Status 고유값]")
print(f"Train: {train['Weight_Status'].unique()}")
print(f"Test: {test['Weight_Status'].unique()}")

# =============================================================================
# 5. 피처 엔지니어링
# =============================================================================
print("\n" + "=" * 60)
print("=== 피처 엔지니어링 ===")
print("=" * 60)

# 5-1. Height를 인치로 통합
train['Height_Total_Inches'] = train['Height(Feet)'] * 12 + train['Height(Remainder_Inches)']
test['Height_Total_Inches'] = test['Height(Feet)'] * 12 + test['Height(Remainder_Inches)']
print("✅ Height_Total_Inches 생성 완료 (Feet * 12 + Inches)")

# 5-2. BMI 계산
# BMI = 체중(kg) / 키(m)^2
# 파운드 → kg: * 0.453592
# 인치 → m: * 0.0254
train['BMI'] = (train['Weight(lb)'] * 0.453592) / ((train['Height_Total_Inches'] * 0.0254) ** 2)
test['BMI'] = (test['Weight(lb)'] * 0.453592) / ((test['Height_Total_Inches'] * 0.0254) ** 2)
print("✅ BMI 생성 완료 (Weight(kg) / Height(m)^2)")

print(f"\n[BMI 통계 - Train]")
print(train['BMI'].describe())

# =============================================================================
# 6. 피처 선택 (5개 → 9개로 확장)
# =============================================================================
print("\n" + "=" * 60)
print("=== 피처 선택 ===")
print("=" * 60)

# 기존 피처 (5개)
# features_old = ['Exercise_Duration', 'Gender', 'BPM', 'Age', 'Weight(lb)']

# 새로운 피처 (9개) - Height_Total_Inches는 BMI 계산에만 사용, 직접 피처로는 미포함
features = [
    'Exercise_Duration',    # 기존 - 운동 시간
    'Gender',               # 기존 - 성별
    'BPM',                  # 기존 - 심박수
    'Age',                  # 기존 - 나이
    'Weight(lb)',           # 기존 - 체중
    'BMI',                  # 신규 - 체질량지수
    'Body_Temperature(F)',  # 신규 - 체온
    'Weight_Status'         # 신규 - 체중 상태
]

print(f"사용할 피처 ({len(features)}개):")
for i, feat in enumerate(features, 1):
    status = "신규" if feat in ['BMI', 'Body_Temperature(F)', 'Weight_Status'] else "기존"
    print(f"  {i}. {feat} [{status}]")

# 피처 추출
X = train[features].copy()
y = train['Calories_Burned'].copy()
X_test = test[features].copy()

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"X_test shape: {X_test.shape}")

# =============================================================================
# 7. 전처리 (인코딩)
# =============================================================================
print("\n" + "=" * 60)
print("=== 전처리 (인코딩) ===")
print("=" * 60)

# 7-1. Gender 인코딩
le_gender = LabelEncoder()
X['Gender'] = le_gender.fit_transform(X['Gender'])
X_test['Gender'] = le_gender.transform(X_test['Gender'])
print(f"✅ Gender 인코딩 완료: {dict(zip(le_gender.classes_, range(len(le_gender.classes_))))}")

# 7-2. Weight_Status 인코딩
le_weight = LabelEncoder()
X['Weight_Status'] = le_weight.fit_transform(X['Weight_Status'])

# test에 train에 없는 값이 있을 경우 대비
test_weight_status_unique = X_test['Weight_Status'].unique()
for label in test_weight_status_unique:
    if label not in le_weight.classes_:
        le_weight.classes_ = np.append(le_weight.classes_, label)
        print(f"⚠️ Test에만 있는 Weight_Status 값 추가: {label}")

X_test['Weight_Status'] = le_weight.transform(X_test['Weight_Status'])
print(f"✅ Weight_Status 인코딩 완료: {dict(zip(le_weight.classes_, range(len(le_weight.classes_))))}")

print(f"\n[인코딩 후 데이터 타입]")
print(X.dtypes)

# =============================================================================
# 8. PolynomialFeatures 변환
# =============================================================================
print("\n" + "=" * 60)
print("=== PolynomialFeatures 변환 ===")
print("=" * 60)

# 다항 피처 생성 (degree=3, interaction_only=True)
poly = PolynomialFeatures(degree=3, interaction_only=True, include_bias=False)

X_poly = poly.fit_transform(X)
X_test_poly = poly.transform(X_test)

print(f"원본 피처 개수: {X.shape[1]}개")
print(f"변환 후 피처 개수: {X_poly.shape[1]}개")
print(f"\n✅ PolynomialFeatures 변환 완료")
print(f"  - degree: 3")
print(f"  - interaction_only: True")
print(f"  - include_bias: False")

# =============================================================================
# 9. 모델 학습
# =============================================================================
print("\n" + "=" * 60)
print("=== 모델 학습 ===")
print("=" * 60)

# 9-1. Linear Regression
lr = LinearRegression()
lr.fit(X_poly, y)
lr_pred_train = lr.predict(X_poly)
lr_rmse_train = np.sqrt(mean_squared_error(y, lr_pred_train))
print(f"✅ LinearRegression 학습 완료")
print(f"   Train RMSE: {lr_rmse_train:.5f}")

# 9-2. Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_poly, y)
ridge_pred_train = ridge.predict(X_poly)
ridge_rmse_train = np.sqrt(mean_squared_error(y, ridge_pred_train))
print(f"\n✅ Ridge (alpha=1.0) 학습 완료")
print(f"   Train RMSE: {ridge_rmse_train:.5f}")

# 9-3. 앙상블 (Train)
ensemble_pred_train = 0.6 * np.round(lr_pred_train) + 0.4 * np.round(ridge_pred_train)
ensemble_rmse_train = np.sqrt(mean_squared_error(y, ensemble_pred_train))
print(f"\n✅ 앙상블 (0.6*LR + 0.4*Ridge) 완료")
print(f"   Train RMSE: {ensemble_rmse_train:.5f}")

# =============================================================================
# 10. Train RMSE 검증 결과 요약
# =============================================================================
print("\n" + "=" * 60)
print("=== 모델 성능 비교 ===")
print("=" * 60)

print(f"\n{'모델':<25} {'Train RMSE':<15}")
print("-" * 40)
print(f"{'LinearRegression':<25} {lr_rmse_train:<15.5f}")
print(f"{'Ridge (alpha=1.0)':<25} {ridge_rmse_train:<15.5f}")
print(f"{'Ensemble (0.6*LR+0.4*R)':<25} {ensemble_rmse_train:<15.5f}")
print("-" * 40)
print(f"{'기존 모델 (5 features)':<25} {'0.10964':<15}")

# =============================================================================
# 11. Test 예측
# =============================================================================
print("\n" + "=" * 60)
print("=== Test 예측 ===")
print("=" * 60)

# 각 모델로 예측
lr_pred_test = lr.predict(X_test_poly)
ridge_pred_test = ridge.predict(X_test_poly)

# 반올림 후 앙상블
lr_pred_test_rounded = np.round(lr_pred_test)
ridge_pred_test_rounded = np.round(ridge_pred_test)
final_pred = 0.6 * lr_pred_test_rounded + 0.4 * ridge_pred_test_rounded

print("✅ Test 예측 완료")
print(f"\n[예측값 통계]")
print(f"  - Min:  {final_pred.min():.1f}")
print(f"  - Max:  {final_pred.max():.1f}")
print(f"  - Mean: {final_pred.mean():.1f}")
print(f"  - Std:  {final_pred.std():.1f}")

# =============================================================================
# 12. Submission 저장
# =============================================================================
print("\n" + "=" * 60)
print("=== Submission 저장 ===")
print("=" * 60)

# submission 파일 생성
submit['Calories_Burned'] = final_pred

# 저장
output_file = OUTPUT_PATH + 'submission_v2.csv'
submit.to_csv(output_file, index=False)

print(f"✅ 저장 완료")
print(f"   저장 경로: {output_file}")

print(f"\n[submission.csv 미리보기 (상위 5개)]")
print(submit.head())

print(f"\n[submission.csv 통계]")
print(submit['Calories_Burned'].describe())

# =============================================================================
# 13. 최종 결과 요약
# =============================================================================
print("\n" + "=" * 60)
print("=== 최종 결과 요약 ===")
print("=" * 60)

print(f"""
📊 데이터:
   - Train: {train.shape[0]:,}개
   - Test: {test.shape[0]:,}개

🔧 피처 엔지니어링:
   - 기존 피처: 5개
   - 추가 피처: BMI, Body_Temperature(F), Weight_Status
   - 최종 피처: {len(features)}개
   - PolynomialFeatures 후: {X_poly.shape[1]}개

📈 모델 성능 (Train RMSE):
   - LinearRegression: {lr_rmse_train:.5f}
   - Ridge (alpha=1.0): {ridge_rmse_train:.5f}
   - Ensemble: {ensemble_rmse_train:.5f}
   - 기존 모델: 0.10964

💾 저장:
   - 파일: submission_v2.csv
   - 경로: {output_file}

🎯 기대 효과:
   - 추가 피처(BMI, 체온, 체중상태)가 칼로리 소모량과 상관관계가 높음
   - 더 풍부한 정보로 예측 정확도 향상 기대
""")

print("=" * 60)
print("✅ 모든 작업이 완료되었습니다!")
print("=" * 60)
