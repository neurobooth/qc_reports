"""
Classes related to creating and saving PDFs.
"""

import fpdf
from PIL import Image
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt

from neurobooth_analysis_tools.data.types import NeuroboothTask, NeuroboothDevice
from neurobooth_reports.output import get_file_descriptor


class Report(fpdf.FPDF):
    """Base class for all reports with common functionality"""
    def __init__(self, *args, **kwargs):
        if 'orientation' not in kwargs:
            kwargs['orientation'] = 'portrait'
        if 'format' not in kwargs:
            kwargs['format'] = 'letter'
        kwargs['unit'] = 'mm'
        super(Report, self).__init__(*args, **kwargs)
        self.default_font()

    def add_figure(self, fig: plt.Figure, close: bool = False, full_width: bool = False) -> None:
        canvas = FigureCanvas(fig)
        canvas.draw()
        img = Image.fromarray(np.asarray(canvas.buffer_rgba()))

        if full_width:
            self.image(img, w=self.epw)
        else:
            self.image(img)

        if close:
            plt.close(fig)

    def default_font(self) -> None:
        self.set_font('Helvetica', size=10)

    def footer_font(self) -> None:
        self.default_font()

    def header_font(self) -> None:
        self.set_font('Helvetica', style='B', size=14)

    def make_footer(self, text: str = '', text_ratio: float = 0.8, horiz_rule: bool = True) -> None:
        if horiz_rule:
            self.set_y(-12)
            self.cell(w=0, border="T")

        self.set_y(-10)
        self.footer_font()

        # Put the given text on the left
        self.set_x(self.l_margin)
        self.cell(w=self.epw * text_ratio, txt=text, align='L')

        # Put page on the right
        page_str = f'Page {self.page_no()} / {{nb}}'
        self.cell(w=self.epw * (1-text_ratio), txt=page_str, align='R')

    def add_page_with_title(self, title: str) -> None:
        self.add_page()
        self.header_font()
        self.cell(w=self.epw, txt=title, align='C')
        self.default_font()
        self.set_y(16)

    def output_file(self, path: str) -> None:
        with open(get_file_descriptor(path), 'wb') as f:
            f.write(self.output(dest='S'))


class SessionReport(Report):
    """A potentially long report generated for a single session."""
    def __init__(self, session_id: str, *args, **kwargs):
        super(SessionReport, self).__init__(*args, **kwargs)
        self.session_id = session_id

    def footer(self) -> None:
        self.make_footer(f'Session {self.session_id}')


class TaskReport(Report):
    """A potentially long report generated for a single task."""
    def __init__(self, session_id: str, task: NeuroboothTask, *args, **kwargs):
        super(TaskReport, self).__init__(*args, **kwargs)
        self.session_id = session_id
        self.task = task

    def add_device_page(self, device: NeuroboothDevice, device_info: str):
        header_str = device.name
        if device == NeuroboothDevice.RealSense:
            header_str = device_info
        elif device == NeuroboothDevice.Mbient:
            header_str = f'{header_str} ({device_info})'
        self.add_page_with_title(header_str)

    def footer(self) -> None:
        self.make_footer(f'Session {self.session_id}  ({self.task.name})')
