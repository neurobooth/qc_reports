"""
Classes related to creating and saving PDFs.
"""

import fpdf
from PIL import Image
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
from neurobooth_analysis_tools.data.types import NeuroboothTask, NeuroboothDevice


class Report(fpdf.FPDF):
    """Base class for all reports with common functionality"""
    def __init__(self, *args, **kwargs):
        if 'orientation' not in kwargs:
            kwargs['orientation'] = 'portrait'
        if 'format' not in kwargs:
            kwargs['format'] = 'letter'
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

    def default_font(self):
        self.set_font('Helvetica', size=10)

    def footer_font(self):
        self.default_font()

    def header_font(self):
        self.set_font('Helvetica', style='B', size=14)


class SessionReport(Report):
    """A potentially long report generated for a single session."""
    def __init__(self, session_id: str, *args, **kwargs):
        super(SessionReport, self).__init__(*args, **kwargs)
        self.session_id = session_id

    def footer(self):
        self.set_y(-15)
        self.set_x(self.l_margin)
        self.cell(txt=f'Session f{self.session_id}', align='L')
        self.set_x(self.l_margin)
        self.cell(txt=f'Page {self.page_no()}/{{nb}}', align='R')


class TaskReport(Report):
    """A potentially long report generated for a single task."""
    def __init__(self, session_id: str, task: NeuroboothTask, *args, **kwargs):
        super(TaskReport, self).__init__(*args, **kwargs)
        self.session_id = session_id
        self.task = task

    def add_device_page(self, device: NeuroboothDevice, device_info: str):
        self.add_page()
        self.header_font()
        device_info = f' ({device_info})' if device_info else ''
        self.cell(w=self.epw, txt=f'{device.name}{device_info}', align='C')

    def footer(self):
        # Horizontal Rule
        self.set_y(-12)
        self.cell(w=0, border="T")

        self.set_y(-10)
        self.footer_font()
        margin_width = self.r_margin - self.l_margin

        # Session/task info on the left
        info_str = f'Session {self.session_id}  ({self.task.name})'
        self.set_x(self.l_margin)
        self.cell(w=margin_width/2, txt=info_str, align='L')

        # Page info on the right
        page_str = f'Page {self.page_no()} / {{nb}}'
        self.cell(w=margin_width/2, txt=page_str, align='R')
