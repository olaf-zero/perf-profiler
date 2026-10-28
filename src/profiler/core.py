from time import perf_counter
from contextlib import contextmanager
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

WARMUP_PERCENTAGE = 0.05

# Styles
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

class Profiler:
    def __init__(self):
        self.saved_records = defaultdict(list)

    @contextmanager
    def measure(self, name: str):
        start_time = perf_counter()

        try:
            yield
        finally:
            end_time = perf_counter()
            
            time = end_time - start_time
            self.saved_records[name].append(time)

    def print_report(self) -> None:
        for measurement_name in self.saved_records:
            valid_index = int(WARMUP_PERCENTAGE * len(self.saved_records[measurement_name]))
            valid_records = self.saved_records[measurement_name][valid_index:]

            if not valid_records:
                continue

            average     = np.average(valid_records) * 1000
            p50         = np.median(valid_records) * 1000
            p95         = np.percentile(valid_records, 95) * 1000
            p99         = np.percentile(valid_records, 99) * 1000

            lines = [
                f"Measurement session name: {measurement_name}\n",
                "----------------",
                "Test |  Time, ms",
                "----------------",
                f"AVG  |  {average:.5f}",
                f"P50  |  {p50:.5f}",
                f"P95  |  {p95:.5f}",
                f"P99  |  {p99:.5f}\n"
            ]

            print("\n".join(lines))

    def generate_plot(self, names: list[str] | None = None, skip_warmup: bool = True, style: str = "default"):
        if not names:
            valid_names = list(self.saved_records.keys())
        else:
            valid_names = []
            for n in names:
                if n in self.saved_records:
                    valid_names.append(n)
                else:
                    print(f"Error: could not find {n} in records")

        if not valid_names:
            print("Error while generating plot.")
            return

        if style in STYLES:
            slt_style = STYLES[style]  
        else:
            slt_style = STYLES["default"]

        plt.figure(figsize=(10, 6), clear=True, facecolor=slt_style["base"])

        for i, measurement_name in enumerate(valid_names):
            records = self.saved_records[measurement_name]
            total_samples = len(records)

            if total_samples == 0:
                continue

            valid_index = int(WARMUP_PERCENTAGE * total_samples) if skip_warmup else 0

            x_values = np.arange(valid_index, total_samples)
            y_values = np.asarray(records[valid_index:]) * 1000
            
            plt.plot(x_values, y_values, label=measurement_name, color=slt_style["main"][i % len(slt_style["main"])])

        ax = plt.gca()
        ax.set_facecolor(slt_style["base"])

        ax.spines['left'].set_color(slt_style["text"])
        ax.spines['bottom'].set_color(slt_style["text"])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.tick_params(colors=slt_style["text"])

        plt.xlabel("n", color=slt_style["text"])
        plt.ylabel("Time, ms", color=slt_style["text"])

        plt.legend(loc="best", labelcolor=slt_style["text"], facecolor=slt_style["base"], draggable=True)
        plt.grid(True, linestyle="--", c=slt_style["grid"])
        
        plt.show()
        plt.close()