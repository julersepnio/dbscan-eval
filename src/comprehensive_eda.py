import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def comprehensive_eda(df, target_col=None):
    """
    Perform comprehensive exploratory data analysis.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to analyze
    target_col : str, optional
        Target variable (if supervised learning)
    """
    print("="*70)
    print("COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
    print("="*70)

    # ========================================================================
    print("\n1. DATASET OVERVIEW")
    print("="*70)

    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print(f"\nColumn types:")
    print(df.dtypes.value_counts())

    # ========================================================================
    print("\n2. MISSING VALUES ANALYSIS")
    print("="*70)

    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)

    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(
        'Missing Count', ascending=False
    )

    if len(missing_df) > 0:
        print("\nColumns with missing values:")
        print(missing_df)
    else:
        print("\n✓ No missing values found!")

    # ========================================================================
    print("\n3. NUMERICAL FEATURES SUMMARY")
    print("="*70)

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col and target_col in numerical_cols:
        numerical_cols.remove(target_col)

    if len(numerical_cols) > 0:
        print(f"\nNumerical features: {len(numerical_cols)}")
        print(df[numerical_cols].describe().round(3))

        # Plot distributions
        n_cols = min(3, len(numerical_cols))
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        if n_rows * n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, col in enumerate(numerical_cols[:len(axes)]):
            axes[idx].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'{col}\n(mean={df[col].mean():.2f}, std={df[col].std():.2f})')
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(alpha=0.3)

        # Hide empty subplots
        for idx in range(len(numerical_cols), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.show()

    # ========================================================================
    print("\n4. CATEGORICAL FEATURES SUMMARY")
    print("="*70)

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if len(categorical_cols) > 0:
        print(f"\nCategorical features: {len(categorical_cols)}")

        for col in categorical_cols[:5]:  # Show first 5
            print(f"\n{col}:")
            print(f"  Unique values: {df[col].nunique()}")
            print(f"  Top 5 values:")
            print(df[col].value_counts().head())

    # ========================================================================
    if target_col and target_col in df.columns:
        print("\n5. TARGET VARIABLE ANALYSIS")
        print("="*70)

        print(f"\nTarget: {target_col}")

        if df[target_col].dtype in [np.number]:
            if df[target_col].nunique() <= 10:  # Classification
                print("\nClass distribution:")
                print(df[target_col].value_counts())
                print("\nClass proportions:")
                print(df[target_col].value_counts(normalize=True))

                # Plot
                fig, ax = plt.subplots(figsize=(8, 5))
                df[target_col].value_counts().plot(kind='bar', ax=ax, color='steelblue')
                ax.set_title(f'Target Variable Distribution: {target_col}')
                ax.set_xlabel('Class')
                ax.set_ylabel('Count')
                ax.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.show()
            else:  # Regression
                print("\nTarget statistics:")
                print(df[target_col].describe())

    # ========================================================================
    if len(numerical_cols) >= 2:
        print("\n6. CORRELATION ANALYSIS")
        print("="*70)

        # Compute correlation matrix
        corr_matrix = df[numerical_cols[:10]].corr()  # Limit to 10 features for readability

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()

        # Identify highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))

        if high_corr_pairs:
            print("\n⚠️ Highly correlated features (|r| > 0.7):")
            for feat1, feat2, corr in high_corr_pairs:
                print(f"  {feat1} ↔ {feat2}: {corr:.3f}")
            print("\n  Consider removing one from each pair to reduce multicollinearity")

    print("\n" + "="*70)
    print("✓ EDA COMPLETE")
    print("="*70)