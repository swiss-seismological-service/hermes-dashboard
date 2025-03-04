import numpy as np
import pandas as pd


def p_event_ratio_grid(event_counts: pd.DataFrame,
                       n_simulations: int,
                       background: float) -> pd.DataFrame:

    ratio = ((1 - np.exp(-event_counts.values / n_simulations))
             / background)

    return np.clip(ratio, a_min=1, a_max=None)


def p_event(series: pd.Series,
            n_simulations: int,
            beta: float,
            delta_m_min: float) -> float:
    return np.sum(
        1 - np.power(1 - np.exp(-beta * delta_m_min), series)
    ) / n_simulations


def p_event_extrapolate(eventcounts: pd.DataFrame,
                        n_simulations: int,
                        delta_m_min: float,
                        beta: float) -> pd.DataFrame:

    ps = eventcounts.groupby(['oid', 'starttime'],
                             as_index=False)['event_count'].agg(
        lambda x: p_event(x, n_simulations, beta, delta_m_min)
    )
    ps = pd.DataFrame(ps)

    ps = ps.rename(columns={'event_count': 'p_event'})

    if 'starttime' in ps.columns:
        ps = ps.sort_values(by=['starttime'])

    return ps
