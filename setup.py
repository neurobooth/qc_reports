from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='neurobooth-reports',
    version='0.1',
    description='Reporting tools for Neurobooth.',
    long_description=readme(),
    url='https://github.com/neurobooth/qc_reports',
    author='Neurobooth Team',
    author_emails=['boubre@mgh.harvard.edu', 'spatel@phmi.partners.org'],
    license='BSD 3-Clause License',
    packages=['neurobooth_reports'],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'device_timing_report=neurobooth_reports.scripts.device_timing_report:main',
            'prom_completion_report=neurobooth_reports.scripts.prom_completion_report:main'
        ],
    },
)
