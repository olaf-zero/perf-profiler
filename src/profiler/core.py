from time import perf_counter
import numpy as np

WARMUP_PERCENTAGE = 0.05

class Profiler:
    def __init__(self):
        self.start_time = None
        self.saved_records = []

    def start(self) -> None:
        self.start_time = perf_counter()

    def stop(self) -> None:
        if self.start_time is None:
            print("Error: Timer not started.")
            return

        end_time = perf_counter()

        time = end_time - self.start_time
        self.saved_records.append(time)

        self.start_time = None

    def print_report(self) -> None:
        valid_index = int(WARMUP_PERCENTAGE * len(self.saved_records))
        valid_records = self.saved_records[valid_index:]

        if not valid_records:
            return

        average     = np.average(valid_records) * 1000
        p50         = np.median(valid_records) * 1000
        p95         = np.percentile(valid_records, 95) * 1000
        p99         = np.percentile(valid_records, 99) * 1000

        lines = [
            "Test |  Time, ms",
            "----------------",
            f"AVG  |  {average:.5f}",
            f"P50  |  {p50:.5f}",
            f"P95  |  {p95:.5f}",
            f"P99  |  {p99:.5f}"
        ]

        print("\n".join(lines))

    