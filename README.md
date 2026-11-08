# Perf-profiler
A basic, lightweight code profiler for python based on context manager logic, allowing you to measure execution time of your code and generate reports and plots. Its main premise is usability in my other projects so the features might be quite opinionated. 

## Installation
This project is managed by `uv`. The suggested installation process requires you to have already installed that tool. It allows you to quickly import this project as a library using:
```bash
uv add "git+https://github.com/olaf-zero/perf-profiler"
```
or
```bash
uv pip install "git+https://github.com/olaf-zero/perf-profiler"
```
if you want to do a quick test in a basic virtual environment or without adding or editing `pyproject.toml`.

## Quick-start
Usage of profiler is based on a context manager `measure`. This example code shows how to measure your code, generate a report and a plot.
```python
from profiler import Profiler

profiler = Profiler()

with profiler.measure(name="Full time + data fetch"):
    for _ in range(0, 1000):
        with profiler.measure(name="One iteration"):
            # your code

with profiler.measure(name="Another session"):
    # your code

profiler.print_report() # Prints results in terminal (average, median, p95, p99).
profiler.generate_plot(style="default") # generates a plot (can specify which measurement sessions to plot)
```
### Roadmap
This project is currently in a very early stage of development. I am planning to implement those new features and changes:
 - [ ] Change the basic report to a better looking one,
 - [ ] Allow users to add their own report and plot styles,
 - [ ] Allow writing reports and logs to files.