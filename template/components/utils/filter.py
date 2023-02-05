超级慢的 = ['partial_autocorrelation', 'number_cwt_peaks', 'augmented_dickey_fuller', 'lempel_ziv_complexity']
慢的 = ['agg_linear_trend', 'permutation_entropy']
比较慢的 = ['benford_correlation', 'ar_coefficient', 'change_quantiles']

def filter_settings(settings,level = 3):
    if level >= 1:
        # 超过1小时
        for item in 超级慢的:
            if item in settings:
                settings.pop(item)
    if level >= 2:
        for item in 慢的:
            # 超过半小时
            if item in settings:
                settings.pop(item)
    if level >= 3:
        for item in 比较慢的:
            # 超过5分钟
            if item in settings:
                settings.pop(item)
    return settings

