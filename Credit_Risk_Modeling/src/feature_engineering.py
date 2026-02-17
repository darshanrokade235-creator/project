def add_credit_features(df):
    # Example: Credit utilization ratio = total debt / total credit limit
    if 'total_debt' in df.columns and 'total_credit_limit' in df.columns:
        df['credit_utilization'] = df['total_debt'] / df['total_credit_limit']
    else:
        df['credit_utilization'] = 0.3  # Default placeholder

    # Can add more features: debt-to-income ratio, loans per year, etc.
    if 'income' in df.columns and 'loan_amount' in df.columns:
        df['debt_to_income'] = df['loan_amount'] / df['income']

    return df
