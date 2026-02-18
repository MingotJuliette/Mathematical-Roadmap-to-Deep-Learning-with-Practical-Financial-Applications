import pandas as pd
import matplotlib.pyplot as plt
from Deep_learning_RNN_sentiment.src.sentiment_analysis.main import creat_data, forward_stepwise_ols_train_test
# ----------------------------------------------
# 4. Load sentiment data (statement, projection) 
# ----------------------------------------------
Statement_df = pd.read_pickle("Deep_learning_gold_stock/data/processed/statements_embedding.pkl").iloc[1:, -3:]
Statement_df.index = pd.to_datetime(Statement_df.index)

Projection_df = pd.read_pickle("Deep_learning_gold_stock/data/processed/projections_embedding.pkl").iloc[1:, -3:]
Projection_df.index = pd.to_datetime(Projection_df.index)


for index, name in zip(["^MOVE", "GC=F"], ["MOVE", "Gold"]):
    # ------------------------------------------------
    # only keep correlated variable (forward_stepwise)
    # ------------------------------------------------
    sentiment_features = creat_data(Statement_df, Projection_df, index, name)[0].drop(columns= [name,f"{name}_monthly_avg"])
    X = sentiment_features  
    y = creat_data(Statement_df, Projection_df, index, name)[0][f"{name}_monthly_avg"]
    model_final, selected_vars, y_pred = forward_stepwise_ols_train_test(X, y, name, train_frac=0.8, p_enter=0.05)
    print(model_final.summary())
    print(selected_vars)

    merged = creat_data(Statement_df, Projection_df, index, name)[1]
    data = creat_data(Statement_df, Projection_df, index, name)[2]

    # ----------------------------------------
    # Selectino update dates (mid_month)
    # ----------------------------------------

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
    
    # ----------------------------------------
    # Analyse événementielle intra-mensuelle
    # ----------------------------------------

    # ±5 days around publications date
    event_window = 5  # day avant/après
    event_returns = []
    for d in Statement_df.index:
        window = pd.date_range(d - pd.Timedelta(days=event_window),
                            d + pd.Timedelta(days=event_window))
        window_moves = data.loc[data.index.intersection(window), name]
        ret_event = window_moves.pct_change().sum()  # cumulated variance on windows
        event_returns.append((d, ret_event))
    
    event_df = pd.DataFrame(event_returns, columns=["Date", "Return_Event"])

    plt.figure(figsize=(12,5))

    plt.bar(event_df["Date"], event_df["Return_Event"], width=3, linewidth=2, edgecolor="black")

    plt.xlabel("Update date", fontsize=12)
    plt.ylabel(f"Cumulative Return ±{event_window} jour", fontsize=12)
    plt.title(f"Intra-month Impact on {name}", fontsize=14)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)

    ax.tick_params(width=1.5)

    plt.show()
