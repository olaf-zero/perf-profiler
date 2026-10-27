from time import perf_counter
from contextlib import contextmanager
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

WARMUP_PERCENTAGE = 0.05

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

    def generate_plot(self, names: list[str] | None = None, skip_warmup: bool = True):
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
            return

        plt.figure(figsize=(10, 6), clear=True)

        for measurement_name in valid_names:
            records = self.saved_records[measurement_name]
            total_samples = len(records)

            if total_samples == 0:
                continue

            valid_index = int(WARMUP_PERCENTAGE * total_samples) if skip_warmup else 0

            x_values = np.arange(valid_index, total_samples)
            y_values = np.asarray(records[valid_index:]) * 1000
            
            plt.plot(x_values, y_values, label=measurement_name)

        plt.xlabel("n")
        plt.ylabel("Time, ms")

        plt.legend(loc="best")
        plt.grid(True, linestyle="--")
        
        plt.show()
        plt.close()