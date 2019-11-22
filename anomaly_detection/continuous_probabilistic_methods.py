def get_lower_and_upper_bounds(series, method = 'iqr', multiplier = 1.5):
    if method == 'iqr':
        iqr_range = series.quantile(.75)-series.quantile(.25)
        lower_bound = series.quantile(.25) - (iqr_range * multiplier)
        upper_bound = series.quantile(.75) + (iqr_range * multiplier)
    elif method == 'std':
        sigma = series.std()
        mu = series.mean()
        lower_bound = mu - (sigma * multiplier)
        upper_bound = mu + (sigma * multiplier)
    return lower_bound,upper_bound