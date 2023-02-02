import os
import stat
import pandas as pd
import matplotlib.pyplot as plt


def dataframe_to_csv(report_path: str, dataframe: pd.DataFrame) -> None:
    """Write a dataframe to a csv file with common settings. Sets reasonable file permissions."""
    dataframe.to_csv(report_path, index=False, header=True, sep=',')
    set_permissions(report_path)


def save_fig(fig_path: str, fig: plt.Figure, close: bool = True, **kwargs) -> None:
    """Save and optionally close a figure. Sets reasonable file permissions."""
    fig.savefig(fig_path, **kwargs)
    set_permissions(fig_path)
    if close:
        plt.close(fig)


def set_permissions(report_path: str) -> None:
    os.chmod(report_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP)
