from time import perf_counter
from contextlib import contextmanager
import numpy as np
from collections import defaultdict

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

    