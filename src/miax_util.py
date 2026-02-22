import pandas as pd


def calculate_momentum(rebal_date, monthly_closes, eligible_symbols):
    """
    Calcula R_6 y R_12 para cada activo elegible en una fecha de rebalanceo.

    - R_12: suma retornos logarítmicos desde t-13 hasta t-1
    - R_6:  suma retornos logarítmicos desde t-7  hasta t-1
    - Se excluye el mes actual (t) para evitar reversión a la media
    """
    # Definimos los periodos (en meses)
    t_minus_1 = (rebal_date - pd.DateOffset(months=1)).to_period('M')
    t_minus_7 = (rebal_date - pd.DateOffset(months=7)).to_period('M')
    t_minus_13 = (rebal_date - pd.DateOffset(months=13)).to_period('M')

    # Añadimos columna de periodo si no existe
    data = monthly_closes.copy()
    data['period'] = data['date'].dt.to_period('M')

    # Filtramos solo activos elegibles
    data = data[data['symbol'].isin(eligible_symbols)]

    # Ventana R_12: desde t-13 hasta t-1
    mask_12 = (data['period'] >= t_minus_13) & (data['period'] <= t_minus_1)
    r12 = (data[mask_12]
           .groupby('symbol')['log_return']
           .sum()
           .rename('R_12'))

    # Ventana R_6: desde t-7 hasta t-1
    mask_6 = (data['period'] >= t_minus_7) & (data['period'] <= t_minus_1)
    r6 = (data[mask_6]
          .groupby('symbol')['log_return']
          .sum()
          .rename('R_6'))

    return pd.concat([r12, r6], axis=1).dropna()


def calculate_scores(momentum_df):
    """
    Normaliza R_6 y R_12 mediante Z-score dentro del universo
    y calcula el score compuesto. Devuelve el top 20.
    """
    df = momentum_df.copy()

    # Z-scores
    df['Z_12'] = (df['R_12'] - df['R_12'].mean()) / df['R_12'].std()
    df['Z_6'] = (df['R_6'] - df['R_6'].mean()) / df['R_6'].std()

    # Score final: media simple de los dos Z-scores
    df['score'] = (df['Z_12'] + df['Z_6']) / 2

    # Ordenamos y seleccionamos top 20
    top20 = df.sort_values('score', ascending=False).head(20)

    return top20


# ------------------------------------------------------------
# 3. FUNCIÓN DE REBALANCEO
# ------------------------------------------------------------

def rebalance(rebal_date, target_tickers, holdings, cash, parquet, COMMISSION_RATE, COMMISSION_MIN):
    """
    Ejecuta un rebalanceo completo:
    - Vende al OPEN los que salen de la cartera
    - Compra al CLOSE los que entran
    - No toca los que se mantienen
    Devuelve el nuevo estado de holdings, cash y comisiones pagadas
    """
    commissions = 0
    prices_today = (parquet[parquet['date'] == rebal_date]
                    [['symbol', 'open', 'close']]
                    .set_index('symbol'))

    current_tickers = set(holdings.keys())
    target_set = set(target_tickers)

    tickers_to_sell = current_tickers - target_set
    tickers_to_buy = target_set - current_tickers
    tickers_to_keep = current_tickers & target_set

    # -- VENTAS al OPEN --
    for ticker in tickers_to_sell:
        if ticker not in prices_today.index:
            # Sin precio: caso enunciado, vendemos a close del último día disponible
            last_close = (parquet[parquet['symbol'] == ticker]['close'].iloc[-1])
            proceeds = holdings[ticker] * last_close
        else:
            open_price = prices_today.loc[ticker, 'open']
            proceeds = holdings[ticker] * open_price

        commission = max(proceeds * COMMISSION_RATE, COMMISSION_MIN)
        cash += proceeds - commission
        commissions += commission
        del holdings[ticker]

    # -- CAPITAL DISPONIBLE para compras --
    # Reservamos las comisiones estimadas para no quedarnos en negativo
    n_to_buy = len([t for t in tickers_to_buy if t in prices_today.index])

    if n_to_buy > 0:
        # Estimamos comisión por posición y la descontamos del capital disponible
        # commission = position_value * COMMISSION_RATE (asumimos > mínimo)
        # position_value = (cash / n_to_buy) / (1 + COMMISSION_RATE)
        position_value = (cash / n_to_buy) / (1 + COMMISSION_RATE)
    else:
        position_value = 0

    # -- COMPRAS al CLOSE --
    for ticker in tickers_to_buy:
        if ticker not in prices_today.index:
            print(f"⚠️ {ticker} sin precio en {rebal_date}, queda en liquidez")
            continue

        close_price = prices_today.loc[ticker, 'close']
        num_shares = position_value / close_price
        cost = num_shares * close_price
        commission = max(cost * COMMISSION_RATE, COMMISSION_MIN)

        cash -= (cost + commission)
        holdings[ticker] = num_shares
        commissions += commission

    return holdings, cash, commissions