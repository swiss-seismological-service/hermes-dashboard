import matplotlib.cm as cm
import matplotlib.colors as mcolors


def matplotlib_to_plotly(cmap_name, n_colors=10):
    """
    Convert a Matplotlib colormap to a Plotly-compatible colorscale.

    Parameters:
        cmap_name (str or Colormap): Name of the Matplotlib
                                     colormap or a colormap object.
        n_colors (int): Number of discrete colors to sample from the colormap.

    Returns:
        List of (position, color) tuples compatible with Plotly.
    """
    if isinstance(cmap_name, str):
        cmap = cm.get_cmap(cmap_name)  # Get the colormap
    else:
        cmap = cmap_name  # Assume it's already a Colormap object

    colorscale = [
        (i / (n_colors - 1), mcolors.to_hex(cmap(i / (n_colors - 1))))
        for i in range(n_colors)
    ]

    return colorscale
