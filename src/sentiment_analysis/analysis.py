import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm


# 1. FEATURE ENGINEERING

def metrics(data, corr_threshold=0.2, target_col="MOVE_monthly_avg"):
    df = data.copy()

    # 1.a NLP deltas & interaction features
    for doc_type in ["Statement", "Projection"]:
        df[f"{doc_type}_ΔHawkishness"] = df[f"FOMC_{doc_type}_Hawkishness"].diff()
        df[f"{doc_type}_ΔDocShift"] = df[f"FOMC_{doc_type}_DocShift"].diff()
        df[f"{doc_type}_ΔIntraDocVar"] = df[f"FOMC_{doc_type}_IntraDocVar"].diff()
        df = df.dropna(subset=[f"{doc_type}_ΔHawkishness",
                                f"{doc_type}_ΔDocShift",
                                f"{doc_type}_ΔIntraDocVar"])
        # Interaction
        df[f"{doc_type}_ΔHawkishness_x_ΔDocShift"] = \
            df[f"{doc_type}_ΔHawkishness"] * df[f"{doc_type}_ΔDocShift"]

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
            if best_pval is None or pval < best_pval:
                best_pval = pval
                best_var = var
        if best_pval is not None and best_pval < p_enter:
            selected_vars.append(best_var)
            remaining_vars.remove(best_var)
        else:
            break
    
    print("Variables retenues:", selected_vars)
    
    # TEST SET
    X_test = X.iloc[n_train:].copy()
    y_test = y.iloc[n_train:]
    
    X_train_model = sm.add_constant(X_train[selected_vars], has_constant='add')
    X_test_model = sm.add_constant(X_test[selected_vars], has_constant='add')
    
    model_final = sm.OLS(y_train, X_train_model).fit()
    y_pred = model_final.predict(X_test_model)
    
    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(y.index, y.values, label=f"{name}_monthly_avg", marker='o')
    plt.plot(y_pred.index, y_pred.values, label="Forward Stepwise OLS (test)", marker='x')
    plt.axvline(y.index[n_train], color='red', linestyle='--', label='Train/Test split')
    plt.xlabel("Date")
    plt.ylabel(name)
    plt.title("Forward Stepwise OLS with variables dynamique selection")
    plt.legend()
    plt.show()
    
    return model_final, selected_vars, y_pred

def creat_data(df_1, df_2, index, name) : # GC=F ^MOVE
    data = yf.download(index,
                            start=df_1.index.min(),
                            end=df_1.index.max(),
                            auto_adjust=True)["Close"][index]
    data = data.to_frame(name)

    # Merge sentiment features
    merged = data.join(df_1, how="left").join(df_2, how="left")

    # Monthly average MOVE
    merged["month_key"] = merged.index.to_period("M")
    move_monthly = merged.groupby(merged.index.to_period("M"))[name].mean()
    merged[f"{name}_monthly_avg"] = merged["month_key"].map(move_monthly)
    merged.drop(columns="month_key", inplace=True)
    merged = merged.ffill()

    # 6. SAMPLE SELECTION (MID-MONTH)
    monthly_ref = merged.index.to_period("M").unique()
    monthly_rows = []

    for month in monthly_ref:
        target_day = pd.Timestamp(month.start_time.year, month.start_time.month, 15)
        window = pd.date_range(target_day - pd.Timedelta(days=2),
                            target_day + pd.Timedelta(days=2))
        existing_days = [d for d in window if d in merged.index]
        if existing_days:
            monthly_rows.append(merged.loc[existing_days[0]])

    merged_monthly = pd.DataFrame(monthly_rows)


    # 7. CALCULATE DELTAS AND FILTER FEATURES
    merged_monthly = metrics(merged_monthly, corr_threshold=0.2, target_col=f"{name}_monthly_avg")
    return merged_monthly, merged, data

# 4. LOAD SENTIMENT DATA
Statement_df = pd.read_pickle("data/processed/statements_embedding.pkl").iloc[1:, -3:]
Statement_df.index = pd.to_datetime(Statement_df.index)

Projection_df = pd.read_pickle("data/processed/projections_embedding.pkl").iloc[1:, -3:]
Projection_df.index = pd.to_datetime(Projection_df.index)

# Garder seulement les variables corrélées avec MOVE_monthly_avg (comme déjà fait)

for index, name in zip(["^MOVE", "GC=F"], ["MOVE", "Gold"]):
    sentiment_features = creat_data(Statement_df, Projection_df, index, name)[0].drop(columns= [name,f"{name}_monthly_avg"])
    X = sentiment_features  
    y = creat_data(Statement_df, Projection_df, index, name)[0][f"{name}_monthly_avg"]
    model_final, selected_vars, y_pred = forward_stepwise_ols_train_test(X, y, name, train_frac=0.8, p_enter=0.05)
    print(model_final.summary())
    print(selected_vars)

############################################################################################################

    merged = creat_data(Statement_df, Projection_df, index, name)[1]
    data = creat_data(Statement_df, Projection_df, index, name)[2]
    # --- date selection (mid-month) ---
    monthly_ref = merged.index.to_period("M").unique()
    monthly_rows = []
    for month in monthly_ref:
        target_day = pd.Timestamp(month.start_time.year, month.start_time.month, 15)
        window = pd.date_range(target_day - pd.Timedelta(days=2),
                            target_day + pd.Timedelta(days=2))
        existing_days = [d for d in window if d in merged.index]
        if existing_days:
            monthly_rows.append(merged.loc[existing_days[0]])
    merged_monthly = pd.DataFrame(monthly_rows)


    event_window = 5  # days before/after
    event_returns = []
    for d in Statement_df.index:
        window = pd.date_range(d - pd.Timedelta(days=event_window),
                            d + pd.Timedelta(days=event_window))
        window_moves = data.loc[data.index.intersection(window), name]
        ret_event = window_moves.pct_change().sum() 
        event_returns.append((d, ret_event))

    print(event_returns)
    
    event_df = pd.DataFrame(event_returns, columns=["Date", "Return_Event"])
    plt.figure(figsize=(12,5))
    plt.bar(event_df["Date"], event_df["Return_Event"])
    plt.xlabel("Date d'annonce")
    plt.ylabel("Cumulative Return ±1 jour")
    plt.title(f"Impact intra-mensuel des statements sur {name}")
    plt.show()
