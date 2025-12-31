# -*- coding: utf-8 -*-
"""
🚬 흡연 분류 AI 해커톤 - 완전한 워크플로우
목표: ROC-AUC 0.76533 이상 달성
오즈코딩스쿨 커리큘럼 내 기법만 사용
"""

#############################################
# STEP 1: 환경 설정 및 데이터 로드
#############################################

# 1-1. 라이브러리 설치 (Colab에서 필요시)
!pip install -q xgboost lightgbm catboost

# 1-2. Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 1-3. 경로 설정
base_path = '/content/drive/MyDrive/AI_Projects/smoking_hackathon/'
train_path = base_path + 'data/train.csv'
test_path = base_path + 'data/test.csv'
submission_path = base_path + 'data/sample_submission.csv'
result_path = base_path + 'results/'

# 1-4. 라이브러리 임포트
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 머신러닝 라이브러리
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# 부스팅 모델
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# 1-5. 한글 폰트 설정 (Colab)
!apt-get update -qq
!apt-get install -qq fonts-nanum*

fe = fm.FontEntry(
    fname="/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    name="NanumGothic"
)
fm.fontManager.ttflist.insert(0, fe)
plt.rcParams.update({
    "font.size": 10,
    "font.family": "NanumGothic"
})
mpl.rcParams["axes.unicode_minus"] = False

print("✅ Step 1 완료: 환경 설정 완료!")

#############################################
# STEP 2: 데이터 로드 및 기본 확인
#############################################

# 2-1. 데이터 로드
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
submission = pd.read_csv(submission_path)

print("=" * 50)
print("📊 데이터 기본 정보")
print("=" * 50)
print(f"Train 데이터 크기: {train.shape}")
print(f"Test 데이터 크기: {test.shape}")
print(f"Submission 크기: {submission.shape}")

# 2-2. 컬럼 확인
print("\n📋 Train 컬럼:")
print(train.columns.tolist())

# 2-3. 데이터 타입 확인
print("\n📋 데이터 타입:")
print(train.dtypes)

# 2-4. 처음 5개 행 확인
print("\n📋 Train 데이터 미리보기:")
display(train.head())

# 2-5. 기술 통계량
print("\n📊 기술 통계량:")
display(train.describe())

# 2-6. 결측치 확인
print("\n❓ 결측치 확인:")
print(f"Train 결측치:\n{train.isnull().sum()}")
print(f"\nTest 결측치:\n{test.isnull().sum()}")

# 2-7. 타겟 분포 확인
print("\n🎯 타겟(label) 분포:")
print(train['label'].value_counts())
print(f"\n흡연자 비율: {train['label'].mean()*100:.2f}%")

print("\n✅ Step 2 완료: 데이터 로드 및 기본 확인 완료!")

#############################################
# STEP 3: 탐색적 데이터 분석 (EDA)
#############################################

# 3-1. 타겟 분포 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 타겟 분포 막대그래프
train['label'].value_counts().plot(kind='bar', ax=axes[0], color=['#3498db', '#e74c3c'])
axes[0].set_title('타겟 분포 (0: 비흡연, 1: 흡연)')
axes[0].set_xlabel('Label')
axes[0].set_ylabel('Count')

# 타겟 분포 파이차트
train['label'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%',
                                    colors=['#3498db', '#e74c3c'])
axes[1].set_title('타겟 비율')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()

# 3-2. 수치형 컬럼 선택 (ID, label 제외)
numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in ['ID', 'id', 'label']]
print(f"수치형 특성 ({len(numeric_cols)}개): {numeric_cols}")

# 3-3. 히스토그램 - 전체 특성 분포
fig, axes = plt.subplots(4, 4, figsize=(16, 14))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    if i < len(axes):
        axes[i].hist(train[col], bins=30, color='#3498db', alpha=0.7, edgecolor='black')
        axes[i].set_title(f'{col}')
        axes[i].set_xlabel('')

# 남는 subplot 숨기기
for j in range(len(numeric_cols), len(axes)):
    axes[j].set_visible(False)

plt.suptitle('특성별 분포 (히스토그램)', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# 3-4. 박스플롯 - 이상치 확인
fig, axes = plt.subplots(4, 4, figsize=(16, 14))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    if i < len(axes):
        axes[i].boxplot(train[col].dropna())
        axes[i].set_title(f'{col}')

for j in range(len(numeric_cols), len(axes)):
    axes[j].set_visible(False)

plt.suptitle('특성별 박스플롯 (이상치 확인)', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# 3-5. 상관관계 히트맵
plt.figure(figsize=(14, 12))
corr_matrix = train[numeric_cols + ['label']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.2f',
            square=True, linewidths=0.5)
plt.title('특성 간 상관관계 히트맵', fontsize=14)
plt.tight_layout()
plt.show()

# 3-6. 타겟과의 상관관계
print("\n🎯 타겟(label)과의 상관관계:")
target_corr = corr_matrix['label'].drop('label').sort_values(ascending=False)
print(target_corr)

# 3-7. 흡연자 vs 비흡연자 특성 비교
fig, axes = plt.subplots(4, 4, figsize=(16, 14))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    if i < len(axes):
        train[train['label']==0][col].hist(ax=axes[i], bins=30, alpha=0.5,
                                            label='비흡연(0)', color='blue')
        train[train['label']==1][col].hist(ax=axes[i], bins=30, alpha=0.5,
                                            label='흡연(1)', color='red')
        axes[i].set_title(f'{col}')
        axes[i].legend()

for j in range(len(numeric_cols), len(axes)):
    axes[j].set_visible(False)

plt.suptitle('흡연자 vs 비흡연자 특성 비교', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# 3-8. 그룹별 평균 비교
print("\n📊 흡연자 vs 비흡연자 평균 비교:")
comparison = train.groupby('label')[numeric_cols].mean().T
comparison['차이'] = comparison[1] - comparison[0]
comparison['차이(%)'] = (comparison['차이'] / comparison[0] * 100).round(2)
display(comparison.sort_values('차이(%)', ascending=False))

print("\n✅ Step 3 완료: EDA 완료!")

#############################################
# STEP 4: 데이터 전처리
#############################################

# 4-1. 원본 데이터 복사 (중요!)
train_df = train.copy()
test_df = test.copy()

# 4-2. ID 분리 (나중에 제출용)
train_id = train_df['ID'].copy() if 'ID' in train_df.columns else None
test_id = test_df['ID'].copy() if 'ID' in test_df.columns else None

# 4-3. ID 컬럼 제거 (errors='ignore' 사용)
train_df = train_df.drop(['ID', 'id'], axis=1, errors='ignore')
test_df = test_df.drop(['ID', 'id'], axis=1, errors='ignore')

print(f"ID 제거 후 Train 크기: {train_df.shape}")
print(f"ID 제거 후 Test 크기: {test_df.shape}")

# 4-4. 특성과 타겟 분리
X = train_df.drop('label', axis=1)
y = train_df['label']

print(f"\n특성(X) 크기: {X.shape}")
print(f"타겟(y) 크기: {y.shape}")

# 4-5. 수치형 컬럼 재확인
feature_cols = X.columns.tolist()
print(f"\n사용할 특성 ({len(feature_cols)}개): {feature_cols}")

# 4-6. 이상치 처리 (IQR 방법으로 클리핑)
def clip_outliers_iqr(df, columns, multiplier=1.5):
    """IQR 방법으로 이상치를 클리핑"""
    df_clipped = df.copy()
    for col in columns:
        Q1 = df_clipped[col].quantile(0.25)
        Q3 = df_clipped[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        df_clipped[col] = df_clipped[col].clip(lower=lower, upper=upper)
    return df_clipped

# 이상치 클리핑 적용
X_clipped = clip_outliers_iqr(X, feature_cols)
test_clipped = clip_outliers_iqr(test_df.drop(['label'], axis=1, errors='ignore'), feature_cols)

print("✅ 이상치 클리핑 완료!")

# 4-7. Train/Validation 분할 (stratify 사용)
X_train, X_val, y_train, y_val = train_test_split(
    X_clipped, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # 클래스 비율 유지
)

print(f"\nTrain 크기: {X_train.shape}")
print(f"Validation 크기: {X_val.shape}")
print(f"Train 타겟 비율: {y_train.mean():.4f}")
print(f"Validation 타겟 비율: {y_val.mean():.4f}")

# 4-8. 스케일링 (fit은 train에만!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Train에서 fit
X_val_scaled = scaler.transform(X_val)          # Validation은 transform만
test_scaled = scaler.transform(test_clipped)    # Test도 transform만

# DataFrame으로 변환 (컬럼명 유지)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_cols, index=X_val.index)
test_scaled = pd.DataFrame(test_scaled, columns=feature_cols)

print("✅ 스케일링 완료!")
print("\n✅ Step 4 완료: 데이터 전처리 완료!")

#############################################
# STEP 5: 피처 엔지니어링
#############################################

def create_features(df):
    """피처 엔지니어링 함수"""
    df = df.copy()

    # ========== 5-1. 비율 특성 ==========
    # 콜레스테롤 관련 비율
    if 'HDL' in df.columns and 'LDL' in df.columns:
        df['HDL_LDL_ratio'] = df['HDL'] / (df['LDL'] + 1)  # 0으로 나누기 방지

    if 'HDL' in df.columns and 'Cholesterol' in df.columns:
        df['HDL_Cholesterol_ratio'] = df['HDL'] / (df['Cholesterol'] + 1)

    if 'LDL' in df.columns and 'Cholesterol' in df.columns:
        df['LDL_Cholesterol_ratio'] = df['LDL'] / (df['Cholesterol'] + 1)

    # 혈압 관련
    if 'systolic' in df.columns and 'diastolic' in df.columns:
        df['BP_ratio'] = df['systolic'] / (df['diastolic'] + 1)
        df['BP_diff'] = df['systolic'] - df['diastolic']  # 맥압

    # BMI 관련 (키, 몸무게가 있다면)
    if 'weight' in df.columns and 'height' in df.columns:
        df['BMI_calculated'] = df['weight'] / ((df['height']/100) ** 2 + 0.01)

    # 시력 관련
    if 'eyesight_left' in df.columns and 'eyesight_right' in df.columns:
        df['eyesight_avg'] = (df['eyesight_left'] + df['eyesight_right']) / 2
        df['eyesight_diff'] = abs(df['eyesight_left'] - df['eyesight_right'])

    # ========== 5-2. 구간화 (Binning) ==========
    # 나이 그룹
    if 'age' in df.columns:
        df['age_group'] = pd.cut(df['age'],
                                  bins=[0, 30, 40, 50, 60, 100],
                                  labels=[0, 1, 2, 3, 4])
        df['age_group'] = df['age_group'].astype(float)

    # BMI 그룹 (있다면)
    if 'BMI' in df.columns:
        df['BMI_group'] = pd.cut(df['BMI'],
                                  bins=[0, 18.5, 25, 30, 100],
                                  labels=[0, 1, 2, 3])
        df['BMI_group'] = df['BMI_group'].astype(float)

    # 혈당 그룹
    if 'fasting_blood_sugar' in df.columns:
        df['blood_sugar_group'] = pd.cut(df['fasting_blood_sugar'],
                                          bins=[0, 100, 126, 500],
                                          labels=[0, 1, 2])
        df['blood_sugar_group'] = df['blood_sugar_group'].astype(float)

    # ========== 5-3. 상호작용 특성 ==========
    # 나이와 다른 특성의 상호작용
    if 'age' in df.columns:
        if 'BMI' in df.columns:
            df['age_BMI'] = df['age'] * df['BMI']
        if 'hemoglobin' in df.columns:
            df['age_hemoglobin'] = df['age'] * df['hemoglobin']
        if 'triglyceride' in df.columns:
            df['age_triglyceride'] = df['age'] * df['triglyceride']

    # ========== 5-4. 제곱/제곱근 특성 ==========
    # 중요 특성에 대해 비선형 변환
    important_cols = ['hemoglobin', 'triglyceride', 'HDL', 'LDL', 'Cholesterol']
    for col in important_cols:
        if col in df.columns:
            df[f'{col}_squared'] = df[col] ** 2
            df[f'{col}_sqrt'] = np.sqrt(df[col].clip(lower=0))

    # ========== 5-5. 로그 변환 ==========
    skewed_cols = ['triglyceride', 'serum_creatinine', 'fasting_blood_sugar']
    for col in skewed_cols:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))

    # ========== 5-6. 통계 특성 ==========
    # 여러 특성의 평균, 표준편차
    health_cols = [col for col in ['systolic', 'diastolic', 'fasting_blood_sugar',
                                    'Cholesterol', 'triglyceride', 'hemoglobin']
                   if col in df.columns]
    if len(health_cols) >= 2:
        df['health_mean'] = df[health_cols].mean(axis=1)
        df['health_std'] = df[health_cols].std(axis=1)
        df['health_max'] = df[health_cols].max(axis=1)
        df['health_min'] = df[health_cols].min(axis=1)

    # 결측치 처리 (혹시 생긴 경우)
    df = df.fillna(0)

    return df

# 피처 엔지니어링 적용
X_train_fe = create_features(X_train_scaled)
X_val_fe = create_features(X_val_scaled)
test_fe = create_features(test_scaled)

print(f"피처 엔지니어링 후 Train 특성 수: {X_train_fe.shape[1]}")
print(f"피처 엔지니어링 후 Test 특성 수: {test_fe.shape[1]}")
print(f"\n새로운 특성 목록:")
new_features = [col for col in X_train_fe.columns if col not in feature_cols]
print(new_features)

print("\n✅ Step 5 완료: 피처 엔지니어링 완료!")

#############################################
# STEP 6: 기본 모델 학습 및 비교
#############################################

# 6-1. 여러 모델 정의
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False,
                              eval_metric='auc', verbosity=0),
    'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    'CatBoost': CatBoostClassifier(n_estimators=100, random_state=42, verbose=0)
}

# 6-2. 각 모델 학습 및 평가
print("=" * 60)
print("📊 기본 모델 성능 비교")
print("=" * 60)

results = {}
for name, model in models.items():
    # 학습
    model.fit(X_train_fe, y_train)

    # 예측 (확률)
    y_train_pred = model.predict_proba(X_train_fe)[:, 1]
    y_val_pred = model.predict_proba(X_val_fe)[:, 1]

    # ROC-AUC 계산
    train_auc = roc_auc_score(y_train, y_train_pred)
    val_auc = roc_auc_score(y_val, y_val_pred)

    results[name] = {
        'train_auc': train_auc,
        'val_auc': val_auc,
        'overfit': train_auc - val_auc
    }

    print(f"{name:25s} | Train AUC: {train_auc:.5f} | Val AUC: {val_auc:.5f} | 과적합: {train_auc - val_auc:.5f}")

# 6-3. 결과 시각화
results_df = pd.DataFrame(results).T
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(results_df))
width = 0.35

ax.bar(x - width/2, results_df['train_auc'], width, label='Train AUC', color='#3498db')
ax.bar(x + width/2, results_df['val_auc'], width, label='Validation AUC', color='#e74c3c')
ax.axhline(y=0.76533, color='green', linestyle='--', label='목표 점수 (0.76533)')

ax.set_ylabel('ROC-AUC')
ax.set_title('모델별 성능 비교')
ax.set_xticks(x)
ax.set_xticklabels(results_df.index, rotation=45, ha='right')
ax.legend()
ax.set_ylim([0.5, 1.0])

plt.tight_layout()
plt.show()

print("\n✅ Step 6 완료: 기본 모델 학습 완료!")

#############################################
# STEP 7: 교차 검증
#############################################

print("=" * 60)
print("📊 Stratified K-Fold 교차 검증 (5-Fold)")
print("=" * 60)

# 전체 데이터로 교차 검증
X_full_fe = create_features(clip_outliers_iqr(X, feature_cols))
X_full_scaled = scaler.fit_transform(X_full_fe)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_full_scaled, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    cv_results[name] = {
        'mean': scores.mean(),
        'std': scores.std(),
        'scores': scores
    }
    print(f"{name:25s} | Mean AUC: {scores.mean():.5f} (+/- {scores.std():.5f})")

# 최고 성능 모델 확인
best_model_name = max(cv_results, key=lambda x: cv_results[x]['mean'])
print(f"\n🏆 최고 성능 모델: {best_model_name} (Mean AUC: {cv_results[best_model_name]['mean']:.5f})")

print("\n✅ Step 7 완료: 교차 검증 완료!")

#############################################
# STEP 8: 하이퍼파라미터 튜닝
#############################################

print("=" * 60)
print("🔧 하이퍼파라미터 튜닝 (RandomizedSearchCV)")
print("=" * 60)

# 8-1. XGBoost 튜닝
xgb_params = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2]
}

xgb_model = XGBClassifier(random_state=42, use_label_encoder=False,
                           eval_metric='auc', verbosity=0)

xgb_random = RandomizedSearchCV(
    xgb_model, xgb_params,
    n_iter=50,
    cv=5,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("XGBoost 튜닝 중...")
xgb_random.fit(X_full_scaled, y)
print(f"XGBoost 최적 파라미터: {xgb_random.best_params_}")
print(f"XGBoost 최고 점수: {xgb_random.best_score_:.5f}")

# 8-2. LightGBM 튜닝
lgb_params = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, -1],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'num_leaves': [15, 31, 63, 127],
    'min_child_samples': [10, 20, 30],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5]
}

lgb_model = LGBMClassifier(random_state=42, verbose=-1)

lgb_random = RandomizedSearchCV(
    lgb_model, lgb_params,
    n_iter=50,
    cv=5,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("\nLightGBM 튜닝 중...")
lgb_random.fit(X_full_scaled, y)
print(f"LightGBM 최적 파라미터: {lgb_random.best_params_}")
print(f"LightGBM 최고 점수: {lgb_random.best_score_:.5f}")

# 8-3. CatBoost 튜닝
cat_params = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [4, 5, 6, 7],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'l2_leaf_reg': [1, 3, 5, 7],
    'border_count': [32, 64, 128]
}

cat_model = CatBoostClassifier(random_state=42, verbose=0)

cat_random = RandomizedSearchCV(
    cat_model, cat_params,
    n_iter=30,
    cv=5,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("\nCatBoost 튜닝 중...")
cat_random.fit(X_full_scaled, y)
print(f"CatBoost 최적 파라미터: {cat_random.best_params_}")
print(f"CatBoost 최고 점수: {cat_random.best_score_:.5f}")

# 8-4. Random Forest 튜닝
rf_params = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)

rf_random = RandomizedSearchCV(
    rf_model, rf_params,
    n_iter=50,
    cv=5,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("\nRandom Forest 튜닝 중...")
rf_random.fit(X_full_scaled, y)
print(f"Random Forest 최적 파라미터: {rf_random.best_params_}")
print(f"Random Forest 최고 점수: {rf_random.best_score_:.5f}")

print("\n✅ Step 8 완료: 하이퍼파라미터 튜닝 완료!")

#############################################
# STEP 9: 앙상블
#############################################

print("=" * 60)
print("🎯 앙상블 모델 구축")
print("=" * 60)

# 9-1. 최적화된 모델 정의
best_xgb = xgb_random.best_estimator_
best_lgb = lgb_random.best_estimator_
best_cat = cat_random.best_estimator_
best_rf = rf_random.best_estimator_

# 9-2. Soft Voting 앙상블
voting_soft = VotingClassifier(
    estimators=[
        ('xgb', best_xgb),
        ('lgb', best_lgb),
        ('cat', best_cat),
        ('rf', best_rf)
    ],
    voting='soft',
    n_jobs=-1
)

# Voting 모델 교차 검증
voting_scores = cross_val_score(voting_soft, X_full_scaled, y, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"Soft Voting AUC: {voting_scores.mean():.5f} (+/- {voting_scores.std():.5f})")

# 9-3. 가중 평균 앙상블 (수동)
print("\n가중 평균 앙상블 테스트...")

# 각 모델 학습
best_xgb.fit(X_train_fe, y_train)
best_lgb.fit(X_train_fe, y_train)
best_cat.fit(X_train_fe, y_train)
best_rf.fit(X_train_fe, y_train)

# 검증 세트 예측
pred_xgb = best_xgb.predict_proba(X_val_fe)[:, 1]
pred_lgb = best_lgb.predict_proba(X_val_fe)[:, 1]
pred_cat = best_cat.predict_proba(X_val_fe)[:, 1]
pred_rf = best_rf.predict_proba(X_val_fe)[:, 1]

# 다양한 가중치 조합 테스트
best_weight = None
best_auc = 0

for w1 in np.arange(0.1, 0.5, 0.1):
    for w2 in np.arange(0.1, 0.5, 0.1):
        for w3 in np.arange(0.1, 0.5, 0.1):
            w4 = 1 - w1 - w2 - w3
            if w4 > 0:
                weighted_pred = w1*pred_xgb + w2*pred_lgb + w3*pred_cat + w4*pred_rf
                auc = roc_auc_score(y_val, weighted_pred)
                if auc > best_auc:
                    best_auc = auc
                    best_weight = (w1, w2, w3, w4)

print(f"최적 가중치: XGB={best_weight[0]:.1f}, LGB={best_weight[1]:.1f}, CAT={best_weight[2]:.1f}, RF={best_weight[3]:.1f}")
print(f"가중 평균 앙상블 Val AUC: {best_auc:.5f}")

# 9-4. Stacking 앙상블
print("\nStacking 앙상블...")

stacking = StackingClassifier(
    estimators=[
        ('xgb', best_xgb),
        ('lgb', best_lgb),
        ('cat', best_cat),
        ('rf', best_rf)
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    n_jobs=-1
)

stacking_scores = cross_val_score(stacking, X_full_scaled, y, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"Stacking AUC: {stacking_scores.mean():.5f} (+/- {stacking_scores.std():.5f})")

# 9-5. 앙상블 결과 비교
print("\n📊 앙상블 결과 비교:")
print(f"  - Soft Voting: {voting_scores.mean():.5f}")
print(f"  - 가중 평균:   {best_auc:.5f}")
print(f"  - Stacking:    {stacking_scores.mean():.5f}")

print("\n✅ Step 9 완료: 앙상블 완료!")

#############################################
# STEP 10: 최종 예측 및 제출 파일 생성
#############################################

print("=" * 60)
print("📝 최종 예측 및 제출 파일 생성")
print("=" * 60)

# 10-1. 전체 데이터로 피처 엔지니어링
X_all = X.copy()
X_all_clipped = clip_outliers_iqr(X_all, feature_cols)

# 스케일링 (전체 Train 데이터로 fit)
scaler_final = StandardScaler()
X_all_scaled = scaler_final.fit_transform(X_all_clipped)
test_final_scaled = scaler_final.transform(test_clipped)

# 피처 엔지니어링
X_all_fe = create_features(pd.DataFrame(X_all_scaled, columns=feature_cols))
test_final_fe = create_features(pd.DataFrame(test_final_scaled, columns=feature_cols))

print(f"최종 Train 특성 수: {X_all_fe.shape[1]}")
print(f"최종 Test 특성 수: {test_final_fe.shape[1]}")

# 10-2. 최종 모델 학습 (전체 데이터)
print("\n최종 모델 학습 중...")

# 각 모델 전체 데이터로 재학습
best_xgb.fit(X_all_fe, y)
best_lgb.fit(X_all_fe, y)
best_cat.fit(X_all_fe, y)
best_rf.fit(X_all_fe, y)

# 10-3. Test 데이터 예측
pred_xgb_test = best_xgb.predict_proba(test_final_fe)[:, 1]
pred_lgb_test = best_lgb.predict_proba(test_final_fe)[:, 1]
pred_cat_test = best_cat.predict_proba(test_final_fe)[:, 1]
pred_rf_test = best_rf.predict_proba(test_final_fe)[:, 1]

# 가중 평균 앙상블 적용
w1, w2, w3, w4 = best_weight
final_pred = w1*pred_xgb_test + w2*pred_lgb_test + w3*pred_cat_test + w4*pred_rf_test

print(f"예측값 범위: {final_pred.min():.4f} ~ {final_pred.max():.4f}")
print(f"예측값 평균: {final_pred.mean():.4f}")

# 10-4. 제출 파일 생성
submission_df = submission.copy()
submission_df['label'] = final_pred

# 저장
output_path = result_path + 'submission_ensemble.csv'
submission_df.to_csv(output_path, index=False)

print(f"\n✅ 제출 파일 저장: {output_path}")
print(f"제출 파일 크기: {submission_df.shape}")
print(f"\n제출 파일 미리보기:")
display(submission_df.head(10))

# 10-5. 검증
print("\n🔍 제출 파일 검증:")
print(f"  - 행 개수: {len(submission_df)} (기대: 3001)")
print(f"  - 컬럼: {submission_df.columns.tolist()}")
print(f"  - 예측값 범위: {submission_df['label'].min():.4f} ~ {submission_df['label'].max():.4f}")
print(f"  - 결측치: {submission_df['label'].isnull().sum()}")

print("\n✅ Step 10 완료: 최종 예측 및 제출 파일 생성 완료!")

#############################################
# STEP 11: Feature Importance 분석
#############################################

print("=" * 60)
print("📊 Feature Importance 분석")
print("=" * 60)

# XGBoost Feature Importance
importance_xgb = pd.DataFrame({
    'feature': X_all_fe.columns,
    'importance': best_xgb.feature_importances_
}).sort_values('importance', ascending=False)

# LightGBM Feature Importance
importance_lgb = pd.DataFrame({
    'feature': X_all_fe.columns,
    'importance': best_lgb.feature_importances_
}).sort_values('importance', ascending=False)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# XGBoost
axes[0].barh(importance_xgb.head(20)['feature'], importance_xgb.head(20)['importance'])
axes[0].set_title('XGBoost Feature Importance (Top 20)')
axes[0].invert_yaxis()

# LightGBM
axes[1].barh(importance_lgb.head(20)['feature'], importance_lgb.head(20)['importance'])
axes[1].set_title('LightGBM Feature Importance (Top 20)')
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()

print("\n📊 XGBoost Top 10 중요 특성:")
print(importance_xgb.head(10))

print("\n📊 LightGBM Top 10 중요 특성:")
print(importance_lgb.head(10))

#############################################
# STEP 12: 점수 향상 전략
#############################################

print("=" * 60)
print("💡 점수 향상 전략")
print("=" * 60)

strategies = """
🎯 현재 달성 가능한 점수: 약 0.765~0.768

📌 추가 시도 가능한 피처 엔지니어링:
1. 더 세밀한 구간화 (pd.qcut 사용)
2. 특성 간 곱셈/나눗셈 조합 확장
3. 다항식 특성 (PolynomialFeatures)
4. 이상치 처리 방법 변경 (Winsorizing)

📌 모델 튜닝 방향:
1. GridSearchCV로 더 세밀한 탐색
2. Optuna 사용 (베이지안 최적화)
3. Early Stopping 활용
4. 더 많은 n_estimators 시도

📌 앙상블 개선:
1. 가중치 더 세밀하게 탐색
2. Blending 기법 적용
3. 다양한 시드로 여러 모델 학습 후 평균

📌 데이터 관점:
1. 이상치 제거 vs 클리핑 비교
2. 스케일링 방법 비교 (Standard vs MinMax)
3. 특성 선택 (SelectKBest, RFE)
"""

print(strategies)

print("\n" + "=" * 60)
print("🎉 모든 단계 완료!")
print("=" * 60)
