import pandas as pd

# ------------------------------------------------------------
# Metodos auxiliares Notebook 3_strategy
# ------------------------------------------------------------

def z_score(r12_df, r6_df):
    """
    Normaliza R_6 y R_12 mediante Z-score dentro del universo
    y calcula el score compuesto. Devuelve el top 20.
    """

    # Z-score de R_12
    z12 = (r12_df - r12_df.mean()) / r12_df.std()
    z12.name = 'Z_12'

    # Z-score de R_6
    z6 = (r6_df - r6_df.mean()) / r6_df.std()
    z6.name = 'Z_6'

    return pd.concat([z12, z6], axis=1)


def top20(z_scores):
    """
    Calcula el score compuesto como media de Z_12 y Z_6,
    ordena de mayor a menor y devuelve el top 20.
    """

    # Score final: media simple de los dos Z-scores
    score = (z_scores['Z_12'] + z_scores['Z_6']) / 2
    score.name = 'score'
    # Ordenamos y seleccionamos top 20
    return score.sort_values(ascending=False).head(20)


def calculate_rebalance(rebal_date: str, universe: pd.DataFrame, monthly_closes):
    """
    Calcula R_6 y R_12 para cada activo elegible en una fecha de rebalanceo.

    - R_12: suma retornos logarítmicos desde t-13 hasta t-1
    - R_6:  suma retornos logarítmicos desde t-7  hasta t-1
    - Se excluye el mes actual (t) para evitar reversion a la media
    """

    eligible_symbols = universe[universe['rebal_date'] == rebal_date]['symbol'].tolist()

    end = (rebal_date - pd.DateOffset(months=1)).to_period('M')
    r6_start = (rebal_date - pd.DateOffset(months=7)).to_period('M')
    r12_start = (rebal_date - pd.DateOffset(months=13)).to_period('M')

    # Filtramos activos elegibles dentro de la ventana R_12
    eligible_symbol_closes = monthly_closes[monthly_closes['symbol'].isin(eligible_symbols)]
    # Ponemos las fechas en period para no dejar ningun mes fuera porque su ultimo dia bursatil no sea el ultimo del mes
    eligible_symbol_closes['period'] = eligible_symbol_closes['date'].dt.to_period('M')
    mask_6 = (eligible_symbol_closes['period'] >= r6_start) & (eligible_symbol_closes['period'] <= end)
    mask_12 = (eligible_symbol_closes['period'] >= r12_start) & (eligible_symbol_closes['period'] <= end)

    # Obtenemos los datos
    r6 = (eligible_symbol_closes[mask_6].groupby('symbol')['log_return'].sum().rename('R_6'))
    r12 = (eligible_symbol_closes[mask_12].groupby('symbol')['log_return'].sum().rename('R_12'))

    scores = z_score(r12, r6)
    selected_symbols = top20(scores)

    result = pd.concat([r12, r6, scores, selected_symbols], axis=1).loc[selected_symbols.index]
    result['rebal_date'] = rebal_date
    return result


def run_strategy(rebal_dates: pd.DataFrame, universe: pd.DataFrame, monthly_closes):
    results = []
    for _, row in rebal_dates.iterrows():
        date = row['date']
        results.append(calculate_rebalance(date, universe, monthly_closes))

    return pd.concat(results)

# ------------------------------------------------------------
# 3. FUNCIÓN DE REBALANCEO
# ------------------------------------------------------------
def check_and_rebalance(rebal_date, tickers, holdings, cash, parquet, commission_rate, commission_min):
    """
    Ejecuta un rebalanceo completo:
    - Vende al OPEN los que salen de la cartera
    - Compra al CLOSE los que entran
    - No toca los que se mantienen
    Retorna el nuevo estado de holdings, cash y comisiones pagadas
    """
    commissions = 0
    prices_today = (parquet[parquet['date'] == rebal_date][['symbol', 'open', 'close']].set_index('symbol'))

    target_tickers = set(tickers)
    current_tickers = set(holdings.keys())

    tickers_to_sell = current_tickers - target_tickers
    tickers_to_buy  = target_tickers - current_tickers

    # Venta
    for ticker in tickers_to_sell:

        if ticker not in prices_today.index:
            last_close = parquet[parquet['symbol'] == ticker]['close'].iloc[-1]
            proceeds = holdings[ticker] * last_close
            print(f"{ticker} sin precio en {rebal_date}, vendido al último close: {last_close:.2f}")
        else:
            prices = prices_today.loc[ticker]
            sell_price = prices['open'] if pd.notna(prices['open']) else prices['close']
            proceeds = holdings[ticker] * sell_price

        commission = max(proceeds * commission_rate, commission_min)
        cash += proceeds - commission
        commissions += commission
        del holdings[ticker]

    tickers_available_to_buy = [t for t in tickers_to_buy if t in prices_today.index]
    n_to_buy = len(tickers_available_to_buy)
    position_value = (cash / n_to_buy) / (1 + commission_rate) if n_to_buy > 0 else 0

    for ticker in tickers_available_to_buy:
        close_price = prices_today.loc[ticker, 'close']
        num_shares = position_value / close_price
        cost = num_shares * close_price
        commission = max(cost * commission_rate, commission_min)

        cash -= (cost + commission)
        holdings[ticker] = num_shares
        commissions += commission

    return holdings, cash, commissions


def run_backtest(parquet, portfolio, rebalancing_dates, initial_capital, commission_rate, commission_min):
    """
    Ejecuta el backtesting completo de la estrategia de momentum.

    Itera sobre todas las fechas de rebalanceo, ejecutando las operaciones
    de compra y venta correspondientes y registrando el estado de la cartera
    tras cada rebalanceo.

    Args:
        parquet (pd.DataFrame): Datos históricos de precios (open, close) por ticker y fecha.
        portfolio (pd.DataFrame): Selecciones del top 20 por fecha de rebalanceo.
        rebalancing_dates (pd.DataFrame): Fechas de rebalanceo mensuales.
        initial_capital (float): Capital inicial en dólares.
        commission_rate (float): Tasa de comisión por operación (e.g. 0.0023).
        commission_min (float): Comisión mínima por operación en dólares (e.g. 23).

    Returns:
        tuple:
            - holdings_history (dict): Estado de la cartera tras cada rebalanceo.
            - total_commissions (float): Comisiones totales pagadas.
    """
    holdings = {}
    holdings_history = {}
    total_commissions = 0
    cash = initial_capital

    for _, row in rebalancing_dates.iterrows():
        rebal_date = row['date']
        target = portfolio[portfolio['rebal_date'] == rebal_date]['symbol'].tolist()

        holdings, cash, commissions = (
            check_and_rebalance(rebal_date, target, holdings, cash, parquet, commission_rate, commission_min)
        )

        total_commissions += commissions
        holdings_history[rebal_date] = {'holdings': holdings.copy(), 'cash': cash}

    return holdings_history, total_commissions

def calculate_daily_portfolio_value(parquet, holdings_history, initial_capital):
    """
    Calcula el valor de la cartera para cada día hábil del periodo.

    Para cada intervalo entre rebalanceos consecutivos, valora las posiciones
    activas usando el precio de cierre diario del parquet y suma el cash disponible.

    Args:
        parquet (pd.DataFrame): Datos históricos de precios diarios.
        holdings_history (dict): Estado de la cartera tras cada rebalanceo.
        initial_capital (float): Capital inicial en dólares, usado para calcular el retorno.

    Returns:
        pd.DataFrame: DataFrame con columnas date, portfolio_value y return_pct.
    """
    rebal_dates = sorted(holdings_history.keys())
    # Añadimos la fecha máxima del parquet como límite final
    rebal_dates.append(parquet['date'].max())

    portfolio_values = []

    for i in range(len(rebal_dates) - 1):
        current_date = rebal_dates[i]
        next_date = rebal_dates[i + 1]

        active_holdings = holdings_history[current_date]['holdings']
        active_cash = holdings_history[current_date]['cash']

        period_prices = parquet[
            (parquet['date'] >= current_date) &
            (parquet['date'] < next_date) &
            (parquet['symbol'].isin(active_holdings.keys()))
            ][['date', 'symbol', 'close']].copy()

        shares = pd.Series(active_holdings)
        period_prices['value'] = period_prices['close'] * period_prices['symbol'].map(shares)

        daily_values = period_prices.groupby('date')['value'].sum() + active_cash
        portfolio_values.append(daily_values)

    portfolio_daily = pd.concat(portfolio_values).reset_index()
    portfolio_daily.columns = ['date', 'portfolio_value']
    portfolio_daily['return_pct'] = (portfolio_daily['portfolio_value'] / initial_capital - 1) * 100

    return portfolio_daily