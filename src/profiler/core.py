# TODO:
# 1. Add support for custom styles made by users (custom style class?).
# 4. add funtionf for warringins
# 5. Themes for reports

from time import perf_counter

from contextlib import contextmanager
from collections import defaultdict
from dataclasses import dataclass, fields

from rich.table import Table
from rich.console import Console

# Custom Profiler exceptions
from .exceptions import ProfilerError, SessionNotFoundError
from .exceptions import ThemeError, ColorNotFoundError

import numpy as np
import matplotlib.pyplot as plt
import warnings


# Default fraction of records to be not included in final calculations.
WARMUP_PERCENTAGE = 0.05

RENDER_OPTIONS = {
    "basic",    # using regular print statements
    "rich"      # using rich library - might not work on older terminal emulators
}

HEX_CHARS = set("0123456789abcdefABCDEF")


@dataclass
class Theme:
    name: str
    base: str
    text: str
    grid: str
    main: list[str] | str

    def __post_init__(self):
        for field in fields(self):
            name = field.name
            value = getattr(self, name)

            if name == "name":
                if len(value) == 0:
                    raise ThemeError("Theme name must exist.")

                continue

            if name == "main":
                if isinstance(value, str):
                    value = [value]
                    self.main = value
                elif isinstance(value, list):
                    pass
                else:
                    raise ThemeError("Theme.main must be a list of strings or a string.")

                if len(value) == 0:
                    raise ThemeError("List of main colors is empty.")

                for color in value:
                    if not self._check_color(color):
                        raise ColorNotFoundError(f"{color} is invalid")

            else:
                if not self._check_color(value):
                    raise ColorNotFoundError(f"{value} is invalid")


    def _check_color(self, hex_code: str) -> bool:
        if len(hex_code) == 7 and hex_code[0] == "#" and set(hex_code[1:]).issubset(HEX_CHARS):
            return True
        else:
            return False


@dataclass
class SessionData:
    name: str                       # name of the session
    number_of_records: int          # number of all records
    number_of_valid: int            # number of all records decreased by the warmup measurements
    statistics: dict[str, float]    # dictionary of values calculated that represent the session


BUILTIN_THEMES = {
    "default": 
        Theme(
        name="default",
        base="#ffffff", 
        text="#000000", 
        grid="#e0e0e0", 
        main=["#0055ff", "#ff0033", "#00cc44", "#ff9900", "#9900cc", "#00cccc", "#ff0099", "#996600", "#003399", "#cc0000", 
            "#008000", "#e67e22", "#6c5ce7", "#16a085", "#e84393", "#d35400", "#2c3e50", "#27ae60", "#8e44ad", "#f1c40f"]
        ),
    "catppuccin_mocha":
        Theme(
        name="catppuccin_mocha",
        base="#1e1e2e", 
        text="#cdd6f4", 
        grid="#313244", 
        main=["#cba6f7", "#f38ba8", "#fab387", "#a6e3a1", "#89b4fa", "#f5e0dc", "#f5c2e7", "#eba0ac", "#f9e2af", "#94e2d5"]
        )
}


class Profiler:
    """My performance profiler with basic data analysis and visualisation."""
    def __init__(self):
        self.saved_records = defaultdict(list)
        self.builtin_themes = BUILTIN_THEMES
        self.user_themes = {}


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
                f"Render option {render} not found! Printing using default, basic option",
                category=UserWarning,
                stacklevel=2
            )
            slt_render = "basic"

        sessions: list[SessionData] = []

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
                sessions.append(session_data)
        
        if slt_render == "rich":
            if not sessions:
                return

            console = Console()
            console.print(self._get_rich_report_table(sessions))
            

    def generate_plot(self, names: list[str] | None = None, skip_warmup: bool = True, theme: str | Theme = "default") -> None:
        """Generates and displays a line chart of execution times of selected records

        Args:
            names: A list of measurement session names to be plotted. If None, all session will be plotted.
            skip_warmup: if True do not plot the "warm-up" records.
            theme: A string indicating colors used for plotting records (e.g., 'default' or 'catppuccin_mocha')
        """
        valid_names = self._filter_valid_names(names=names)

        if not valid_names:
            raise SessionNotFoundError(f"None of the provided session names: {names} have been found while generating a plot.")

        if isinstance(theme, Theme):
            self.register_theme(theme=theme)
            theme_name = theme.nama
        else:
            theme_name = theme

        all_themes = self.builtin_themes | self.user_themes

        if theme_name not in all_themes.keys():
            warnings.warn(
                f"Theme {theme_name} not found! Generating using default theme",
                category=UserWarning,
                stacklevel=2
            )
            slt_theme = all_themes["default"]
        else:
            slt_theme = all_themes[theme_name]

        plt.figure(figsize=(10, 6), clear=True, facecolor=slt_theme.base)

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
            
            plt.plot(x_values, y_values, label=measurement_name, color=slt_theme.main[i % len(slt_theme.main)])

        ax = plt.gca()

        self._set_plot_theme(ax=ax, slt_theme=slt_theme)
        
        plt.show()
        plt.close()

    
    def register_theme(self, theme: Theme) -> None:
        self.user_themes[theme.name] = theme


    def _get_valid_records(self, name: str, skip_warmup: bool = True) -> np.ndarray:
        records = self.saved_records[name]

        if not records:
            return np.asarray([])

        valid_index = int(WARMUP_PERCENTAGE * len(records)) if skip_warmup else 0

        return np.asarray(records[valid_index:])

    
    def _calculate_stats(self, records: np.ndarray) -> dict[str, float]:
        average = np.average(records) * 1000
        p50, p95, p99 = np.percentile(records, [50, 95, 99]) * 1000
        min_value = np.min(records) * 1000
        max_value = np.max(records) * 1000
        std = np.std(records) * 1000

        ret = {
            "Average": average, 
            "Median": p50, 
            "Min": min_value, 
            "Max": max_value, 
            "P95": p95, 
            "P99": p99,
            "Std Dev": std
        }

        return ret


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


    def _get_rich_report_table(self, sessions: list[SessionData]) -> Table:
        table = Table()

        table.add_column("Name", justify="center")
        table.add_column("Num of records", justify="center")
        table.add_column("Valid records", justify="center")

        for s in sessions[0].statistics:
            table.add_column(f"{s}, ms", justify="right")

        for s in sessions:
            self._add_rich_table_row(table, s)

        return table

    
    def _add_rich_table_row(self, table: Table, session_data: SessionData) -> None:
        session_info = [session_data.name, session_data.number_of_records, session_data.number_of_valid]
        calculated_stats = [f"{value:.5f}" for value in session_data.statistics.values()]

        full_row = map(str, session_info + calculated_stats)

        table.add_row(*full_row)


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


    def _set_plot_theme(self, ax: plt.Axes, slt_theme: Theme) -> None:
        ax.set_facecolor(slt_theme.base)

        # Plot borders styling.
        ax.spines['left'].set_color(slt_theme.text)
        ax.spines['bottom'].set_color(slt_theme.text)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.tick_params(colors=slt_theme.text)

        plt.xlabel("n", color=slt_theme.text)
        plt.ylabel("Time, ms", color=slt_theme.text)

        plt.legend(loc="best", labelcolor=slt_theme.text, facecolor=slt_theme.base, draggable=True)
        plt.grid(True, linestyle="--", c=slt_theme.grid)
