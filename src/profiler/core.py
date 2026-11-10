# TODO:
# 1. Add support for custom styles made by users (custom style class?).
# 3. Change "style" to "theme"
# 4. add funtionf for warringins

from time import perf_counter

from contextlib import contextmanager
from collections import defaultdict
from dataclasses import dataclass

from rich.table import Table
from rich.console import Console

# Custom Profiler exceptions
from .exceptions import ProfilerError, SessionNotFoundError

import numpy as np
import matplotlib.pyplot as plt
import warnings

# Default fraction of records to be not included in final calculations.
WARMUP_PERCENTAGE = 0.05

STYLES = {
    "default": {
        "base": "#ffffff",
        "text": "#000000",
        "grid": "#e0e0e0",
        "main": ["#0055ff", "#ff0033", "#00cc44", "#ff9900", "#9900cc", "#00cccc", "#ff0099", "#996600", "#003399", "#cc0000", 
                 "#008000", "#e67e22", "#6c5ce7", "#16a085", "#e84393", "#d35400", "#2c3e50", "#27ae60", "#8e44ad", "#f1c40f"]
    },
    "catppuccin_mocha": {
        "base": "#1e1e2e",
        "text": "#cdd6f4",
        "grid": "#313244",
        "main": ["#cba6f7", "#f38ba8", "#fab387", "#a6e3a1", "#89b4fa", "#f5e0dc", "#f5c2e7", "#eba0ac", "#f9e2af", "#94e2d5"]
    }
}

RENDER_OPTIONS = {
    "basic",    # using regular print statements
    "rich"      # using rich library - might not work on older terminal emulators
}

@dataclass
class SessionData:
    name: str                   # name of the session
    number_of_records: int      # number of all records
    number_of_valid: int        # number of all records decreased by the warmup measurements
    statistics: dict[str, float]     # dictionary of values calculated that represent the session


class Profiler:
    """My performance profiler with basic data analysis and visualisation."""
    def __init__(self):
        self.saved_records = defaultdict(list)
        self.table = Table()
        self._init_rich_report_table()
        self.console = Console()


    @contextmanager
    def measure(self, name: str):
        """Context manager for measuring code block's execution time.

        Args:
            name: A name to represent a current measurement session.

        Example:
            >>> profiler = Profiler()
            >>> with profiler.measure(name="Multiplication"):
            ...     c = a * b
        """
        start_time = perf_counter()

        try:
            yield
        except Exception:
            raise
        else:
            end_time = perf_counter()
            
            time = end_time - start_time # Zobaczyć czy nie równe zero 
            self.saved_records[name].append(time)


    def print_report(self, names: list[str] | None = None, render: str = "basic") -> None:
        """Prints a table with average, median (P50), P95 and P99 of every recorded measurement session."""
        valid_names = self._filter_valid_names(names=names)

        if not valid_names:
            raise SessionNotFoundError(f"None of the provided session names: {names} have been found while printing report.")

        if render in RENDER_OPTIONS:
            slt_render = render
        else:
            warnings.warn(
                f"Render option {render} not found! Printing using default, basic style",
                category=UserWarning,
                stacklevel=2
            )
            slt_render = "basic"

        for measurement_name in valid_names:
            # Do not include a fraction of measurements samples to prevent noice from caching and JIT compilation
            valid_records = self._get_valid_records(name=measurement_name, skip_warmup= True)
            
            num = len(self.saved_records[measurement_name])
            valid_num = len(valid_records)

            if valid_num == 0:
                warnings.warn(
                    f"Session {measurement_name} has 0 valid records.",
                    category=UserWarning,
                    stacklevel=2
                )
                continue

            calculated_session_stats = self._calculate_stats(records=valid_records)

            session_data = SessionData(measurement_name, num, valid_num, calculated_session_stats)

            if slt_render == "basic":
                self._print_basic_report_table(session_data)
            elif slt_render == "rich":
                self._add_rich_table_row(session_data)
        
        if slt_render == "rich":
            self.console.print(self.table)
            self._reset_rich_table()

    def generate_plot(self, names: list[str] | None = None, skip_warmup: bool = True, style: str = "default") -> None:
        """Generates and displays a line chart of execution times of selected records

        Args:
            names: A list of measurement session names to be plotted. If None, all session will be plotted.
            skip_warmup: if True do not plot the "warm-up" records.
            style: A string indicating colors used for plotting records (e.g., 'default' or 'catppuccin_mocha')
        """
        valid_names = self._filter_valid_names(names=names)

        if not valid_names:
            raise SessionNotFoundError(f"None of the provided session names: {names} have been found while generating a plot.")

        if style in STYLES:
            slt_style = STYLES[style]  
        else:
            warnings.warn(
                f"Style {style} not found! Generating using default style",
                category=UserWarning,
                stacklevel=2
            )
            slt_style = STYLES["default"]

        plt.figure(figsize=(10, 6), clear=True, facecolor=slt_style["base"])

        for i, measurement_name in enumerate(valid_names):
            records = self._get_valid_records(name=measurement_name, skip_warmup=skip_warmup)
            total_samples = len(records)

            if total_samples == 0:
                warnings.warn(
                    f"Session {measurement_name} has 0 valid records.",
                    category=UserWarning,
                    stacklevel=2
                )
                continue

            x_values = np.arange(1, total_samples + 1)
            y_values = np.asarray(records) * 1000
            
            plt.plot(x_values, y_values, label=measurement_name, color=slt_style["main"][i % len(slt_style["main"])])

        ax = plt.gca()

        self._set_plot_style(ax=ax, slt_style=slt_style)
        
        plt.show()
        plt.close()


    def _get_valid_records(self, name: str, skip_warmup: bool = True) -> np.ndarray:
        records = self.saved_records[name]

        if not records:
            return np.asarray([])

        valid_index = int(WARMUP_PERCENTAGE * len(records)) if skip_warmup else 0

        return np.asarray(records[valid_index:])

    
    def _calculate_stats(self, records: np.ndarray) -> dict[str, float]:
        average = np.average(records) * 1000
        p50, p95, p99 = np.percentile(records, [50, 95, 99]) * 1000

        return {"Average": average, "Median": p50, "P95": p95, "P99": p99}


    def _reset_rich_table(self) -> None:
        self.table = Table()
        self._init_rich_report_table()


    def _print_basic_report_table(self, session_data: SessionData) -> None:
        session_info = [
            f"Measurement session name: {session_data.name}",
            f"All records: {session_data.number_of_records}",
            f"Valid records: {session_data.number_of_valid}\n",
        ]

        stats_header = [
            "-----------------",
            "Test  -  Time, ms",
            "-----------------"
        ]

        stats = [f"{key}  -  {value:.5f}" for key, value in session_data.statistics.items()]

        print("\n".join(session_info + stats_header + stats) + "\n\n")


    def _init_rich_report_table(self) -> None:
        # self.table.title("All measurement sessions")

        self.table.add_column("Name", justify="center")
        self.table.add_column("Num of records", justify="center")
        self.table.add_column("Valid records", justify="center")
        self.table.add_column("Average, ms", justify="center")
        self.table.add_column("Median, ms", justify="center")
        self.table.add_column("P95, ms", justify="center")
        self.table.add_column("P99, ms", justify="center")

    
    def _add_rich_table_row(self, session_data: SessionData) -> None:
        session_info = [session_data.name, session_data.number_of_records, session_data.number_of_valid]
        calculated_stats = [f"{value:.5f}" for value in session_data.statistics.values()]

        full_row = map(str, session_info + calculated_stats)

        self.table.add_row(*full_row)


    def _filter_valid_names(self, names: list[str] | None) -> list[str]:
        if not names:
            return list(self.saved_records.keys())

        valid_names = []

        for n in names:
            if n in self.saved_records:
                valid_names.append(n)
                continue

            warnings.warn(
                f"Could not find {n} in records",
                category=UserWarning,
                stacklevel=3
            )

        return valid_names


    def _set_plot_style(self, ax: plt.Axes, slt_style: dict) -> None:
        ax.set_facecolor(slt_style["base"])

        # Plot borders styling.
        ax.spines['left'].set_color(slt_style["text"])
        ax.spines['bottom'].set_color(slt_style["text"])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.tick_params(colors=slt_style["text"])

        plt.xlabel("n", color=slt_style["text"])
        plt.ylabel("Time, ms", color=slt_style["text"])

        plt.legend(loc="best", labelcolor=slt_style["text"], facecolor=slt_style["base"], draggable=True)
        plt.grid(True, linestyle="--", c=slt_style["grid"])
