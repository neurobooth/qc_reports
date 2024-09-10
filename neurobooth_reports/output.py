import os
import stat
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
DEFAULT_MODE = 0o660


def get_file_descriptor(path: str) -> int:
    return os.open(path, flags=DEFAULT_FLAGS, mode=DEFAULT_MODE)


def dataframe_to_csv(report_path: str, dataframe: pd.DataFrame) -> None:
    """Write a dataframe to a csv file with common settings. Sets reasonable file permissions."""
    with open(get_file_descriptor(report_path), 'w') as f:
        dataframe.to_csv(f, index=False, header=True, sep=',')


def save_fig(fig_path: str, fig: plt.Figure, close: bool = True, **kwargs) -> None:
    """Save and optionally close a figure. Sets reasonable file permissions."""
    with open(get_file_descriptor(fig_path), 'wb') as f:
        fig.savefig(f, **kwargs)

    if close:
        plt.close(fig)
