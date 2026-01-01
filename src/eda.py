"""
탐색적 데이터 분석 (EDA) 모듈

데이터 이해 및 시각화를 위한 분석 도구
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List
import warnings

warnings.filterwarnings('ignore')


class ExploratoryDataAnalysis:
    """EDA 분석 클래스"""

    def __init__(self, figsize: tuple = (12, 8)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-whitegrid')

    def basic_info(self, df: pd.DataFrame) -> None:
        """기본 데이터 정보 출력"""
        print("=" * 60)
        print("📊 데이터 기본 정보")
        print("=" * 60)

        print(f"\n📐 Shape: {df.shape[0]} rows × {df.shape[1]} columns")

        print(f"\n📋 데이터 타입:")
        print(df.dtypes)

        print(f"\n📈 기초 통계량:")
        print(df.describe().round(2))

        print(f"\n❓ 결측치:")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("  결측치 없음")

    def correlation_analysis(self, df: pd.DataFrame,
                             target_col: str = 'calories',
                             save_path: Optional[str] = None) -> pd.DataFrame:
        """
        상관관계 분석

        칼로리와 다른 변수들 간의 상관관계를 분석
        """
        print("\n" + "=" * 60)
        print("🔗 상관관계 분석")
        print("=" * 60)

        # 수치형 컬럼만 선택
        numeric_df = df.select_dtypes(include=[np.number])

        # 상관관계 행렬
        corr_matrix = numeric_df.corr()

        # 타겟과의 상관관계 출력
        if target_col in corr_matrix.columns:
            target_corr = corr_matrix[target_col].sort_values(ascending=False)
            print(f"\n📊 '{target_col}'과(와)의 상관관계:")
            for col, corr in target_corr.items():
                if col != target_col:
                    bar = '█' * int(abs(corr) * 20)
                    sign = '+' if corr > 0 else '-'
                    print(f"  {col:30s} {sign}{abs(corr):.3f} {bar}")

        # 히트맵 시각화
        fig, ax = plt.subplots(figsize=self.figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                    cmap='RdYlBu_r', center=0, ax=ax,
                    annot_kws={'size': 8})
        ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n  💾 저장됨: {save_path}")

        plt.close()

        return corr_matrix

    def target_distribution(self, df: pd.DataFrame,
                            target_col: str = 'calories',
                            save_path: Optional[str] = None) -> None:
        """타겟 변수 분포 시각화"""
        print("\n" + "=" * 60)
        print(f"📈 타겟 변수 '{target_col}' 분포 분석")
        print("=" * 60)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 히스토그램
        axes[0].hist(df[target_col], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].axvline(df[target_col].mean(), color='red', linestyle='--', label=f'Mean: {df[target_col].mean():.1f}')
        axes[0].axvline(df[target_col].median(), color='green', linestyle='--', label=f'Median: {df[target_col].median():.1f}')
        axes[0].set_xlabel(target_col)
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution')
        axes[0].legend()

        # 박스플롯
        axes[1].boxplot(df[target_col], vert=True)
        axes[1].set_ylabel(target_col)
        axes[1].set_title('Box Plot')

        # Q-Q 플롯 (정규성 확인)
        from scipy import stats
        stats.probplot(df[target_col], dist="norm", plot=axes[2])
        axes[2].set_title('Q-Q Plot (Normality Check)')

        plt.suptitle(f'Target Variable Analysis: {target_col}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n  💾 저장됨: {save_path}")

        plt.close()

        # 통계 출력
        print(f"\n  평균: {df[target_col].mean():.2f}")
        print(f"  중앙값: {df[target_col].median():.2f}")
        print(f"  표준편차: {df[target_col].std():.2f}")
        print(f"  최소값: {df[target_col].min():.2f}")
        print(f"  최대값: {df[target_col].max():.2f}")
        print(f"  왜도(Skewness): {df[target_col].skew():.3f}")
        print(f"  첨도(Kurtosis): {df[target_col].kurtosis():.3f}")

    def feature_vs_target(self, df: pd.DataFrame,
                          features: List[str],
                          target_col: str = 'calories',
                          save_path: Optional[str] = None) -> None:
        """특성과 타겟 간의 관계 시각화"""
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]

        for idx, feature in enumerate(features):
            if feature in df.columns:
                axes[idx].scatter(df[feature], df[target_col], alpha=0.6, edgecolors='black', linewidth=0.5)

                # 추세선
                z = np.polyfit(df[feature], df[target_col], 1)
                p = np.poly1d(z)
                x_line = np.linspace(df[feature].min(), df[feature].max(), 100)
                axes[idx].plot(x_line, p(x_line), "r--", alpha=0.8, label='Trend')

                axes[idx].set_xlabel(feature)
                axes[idx].set_ylabel(target_col)
                axes[idx].set_title(f'{feature} vs {target_col}')
                axes[idx].legend()

        # 빈 subplot 제거
        for idx in range(len(features), len(axes)):
            fig.delaxes(axes[idx])

        plt.suptitle('Feature vs Target Relationships', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.close()

    def categorical_analysis(self, df: pd.DataFrame,
                             cat_col: str,
                             target_col: str = 'calories',
                             save_path: Optional[str] = None) -> None:
        """범주형 변수별 타겟 분석"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 박스플롯
        df.boxplot(column=target_col, by=cat_col, ax=axes[0])
        axes[0].set_title(f'{target_col} by {cat_col}')
        axes[0].set_xlabel(cat_col)

        # 평균 바 차트
        group_mean = df.groupby(cat_col)[target_col].mean().sort_values()
        group_mean.plot(kind='barh', ax=axes[1], color='steelblue', edgecolor='black')
        axes[1].set_xlabel(f'Mean {target_col}')
        axes[1].set_title(f'Average {target_col} by {cat_col}')

        plt.suptitle(f'Categorical Analysis: {cat_col}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.close()

    def multicollinearity_check(self, df: pd.DataFrame,
                                threshold: float = 0.8) -> List[tuple]:
        """
        다중공선성 체크

        상관계수가 threshold 이상인 변수 쌍을 찾아 경고
        """
        print("\n" + "=" * 60)
        print(f"⚠️ 다중공선성 체크 (threshold: {threshold})")
        print("=" * 60)

        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()

        high_corr_pairs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]

                if abs(corr_val) >= threshold:
                    high_corr_pairs.append((col1, col2, corr_val))

        if high_corr_pairs:
            print("\n  🚨 높은 상관관계 발견 (다중공선성 위험):")
            for col1, col2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
                print(f"    - {col1} ↔ {col2}: {corr:.3f}")
            print("\n  💡 권장사항: 위 변수 쌍 중 하나를 제거하거나 PCA 적용 고려")
        else:
            print("\n  ✅ 심각한 다중공선성 없음")

        return high_corr_pairs

    def run_full_eda(self, df: pd.DataFrame,
                     target_col: str = 'calories',
                     output_dir: str = 'outputs/eda') -> None:
        """전체 EDA 실행"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # 1. 기본 정보
        self.basic_info(df)

        # 2. 타겟 분포
        self.target_distribution(df, target_col,
                                 save_path=f'{output_dir}/target_distribution.png')

        # 3. 상관관계
        self.correlation_analysis(df, target_col,
                                  save_path=f'{output_dir}/correlation_heatmap.png')

        # 4. 다중공선성 체크
        self.multicollinearity_check(df)

        # 5. 주요 특성 vs 타겟
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        if 'user_id' in numeric_cols:
            numeric_cols.remove('user_id')

        self.feature_vs_target(df, numeric_cols[:9], target_col,
                               save_path=f'{output_dir}/feature_vs_target.png')

        print("\n" + "=" * 60)
        print(f"✅ EDA 완료! 결과물: {output_dir}/")
        print("=" * 60)


if __name__ == "__main__":
    from data_preprocessing import load_and_preprocess
    from feature_engineering import engineer_features

    df = load_and_preprocess("data/raw/calories_burned_data.csv")
    df = engineer_features(df)

    eda = ExploratoryDataAnalysis()
    eda.run_full_eda(df)
