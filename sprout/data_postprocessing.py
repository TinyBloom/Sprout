def filter_data(params, source):
    from scipy.ndimage import gaussian_filter

    target = gaussian_filter(source, sigma=params["post_sigma"])

    return target
