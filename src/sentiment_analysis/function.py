import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm

def metrics(data):
    df = data.copy()

    # NLP deltas & interaction features
    for doc_type in ["Statement", "Projection"]:
        df[f"{doc_type}_D_Hawkishness"] = df[f"FOMC_{doc_type}_Hawkishness"].diff()
        df[f"{doc_type}_D_DocShift"] = df[f"FOMC_{doc_type}_DocShift"].diff()
        df[f"{doc_type}_D_IntraDocVar"] = df[f"FOMC_{doc_type}_IntraDocVar"].diff()
        df = df.dropna(subset=[f"{doc_type}_D_Hawkishness",
                                f"{doc_type}_D_DocShift",
                                f"{doc_type}_D_IntraDocVar"])
        # Interaction
        df[f"{doc_type}_D_Hawkishness_x_D_DocShift"] = \
            df[f"{doc_type}_D_Hawkishness"] * df[f"{doc_type}_D_DocShift"]

    return df

def forward_stepwise_ols_train_test(X, y, name, train_frac=0.8, p_enter=0.05):
    n_total = len(X)
    n_train = int(n_total * train_frac)
    
    # TRAIN SET
    X_train = X.iloc[:n_train].copy()
    y_train = y.iloc[:n_train]
    
    remaining_vars = list(X_train.columns)
    selected_vars = []
    
    # Forward stepwise selection
    while len(remaining_vars) > 0:
        best_pval = None
        best_var = None
        for var in remaining_vars:
            X_temp = sm.add_constant(X_train[selected_vars + [var]], has_constant='add')
            model = sm.OLS(y_train, X_temp).fit()
            pval = model.pvalues[var]

            # check the estimator significativity 
            if best_pval is None or pval < best_pval:
                best_pval = pval
                best_var = var
        if best_pval is not None and best_pval < p_enter:
            selected_vars.append(best_var)
            remaining_vars.remove(best_var)
        else:
            break
    
    print("Variables kept:", selected_vars)
    
    X_test = X.iloc[n_train:].copy()
    y_test = y.iloc[n_train:]
    
    X_train_model = sm.add_constant(X_train[selected_vars], has_constant='add')
    X_test_model = sm.add_constant(X_test[selected_vars], has_constant='add')
    
    model_final = sm.OLS(y_train, X_train_model).fit()
    y_pred = model_final.predict(X_test_model)
    
    plt.figure(figsize=(12,6))
    plt.plot(y.index, y.values, label=f"{name}_monthly_avg", marker='o')
    plt.plot(y_pred.index, y_pred.values, label="Forward Stepwise OLS (test)", marker='x')
    plt.axvline(y.index[n_train], color='red', linestyle='--', label='Train/Test split')
    plt.xlabel("Date")
    plt.ylabel(name)
    plt.title("Forward Stepwise OLS with dynamic variables selection")
    plt.legend()
    plt.show()
    
    return model_final, selected_vars, y_pred

def creat_data(df_1, df_2, index, name) : # GC=F  & ^MOVE
    data = yf.download(index, start=df_1.index.min(), end=df_1.index.max(),auto_adjust=True)["Close"][index]
    data = data.to_frame(name)

    # Merge sentiment features
    merged = data.join(df_1, how="left").join(df_2, how="left")

    # Monthly average MOVE
    merged["month_key"] = merged.index.to_period("M") 
    move_monthly = merged.groupby(merged.index.to_period("M"))[name].mean()
    merged[f"{name}_monthly_avg"] = merged["month_key"].map(move_monthly)
    merged.drop(columns="month_key", inplace=True)
    merged = merged.ffill() # replace nan by last value for statement and projection

    # Sample selection (mid-month)
    monthly_ref = merged.index.to_period("M").unique()
    monthly_rows = []

    for month in monthly_ref:
        target_day = pd.Timestamp(month.start_time.year, month.start_time.month, 15) # every 15 of each months
        window = pd.date_range(target_day - pd.Timedelta(days=2),target_day + pd.Timedelta(days=2))
        existing_days = [d for d in window if d in merged.index]
        if existing_days:
            monthly_rows.append(merged.loc[existing_days[0]])

    merged_monthly = pd.DataFrame(monthly_rows)

    # Calculate data and others features
    merged_monthly = metrics(merged_monthly)
    return merged_monthly, merged, data
