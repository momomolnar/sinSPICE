#Random utility functions for supporting the synthesis

import argparse
import contextlib
import sys
import time

import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description='Script Runner')

    parser.add_argument(
        '--ion', required=True, type=str,
        help='Ion to be synthesized'
    )
    parser.add_argument(
        '--ion_level', required=True, type=int,
        help='Ion level to be synthesized'
    )
    parser.add_argument(
        '--lambda0', required=True, type=float,
        help='Central wavelength'
    )
    parser.add_argument(
        '--numWaves', required=True, type=int,
        help='Number of wavelenghts'
    )
    parser.add_argument(
        '--dLambda', required=True, type=float,
        help='Wavelength grid spacing'
    )

    return parser.parse_args()


class ProgressBar:
    def __init__(self, total, label="Progress", width=30, stream=None):
        self.total = total
        self.label = label
        self.width = width
        self.stream = stream or sys.stderr
        self.current = 0
        self.start_time = None
        self._last_percent = None
        self._is_tty = hasattr(self.stream, "isatty") and self.stream.isatty()

    def __enter__(self):
        self.start_time = time.time()
        self._draw(force=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None and self.current < self.total:
            self.current = self.total
            self._draw(force=True)
        self.stream.write("\n")
        self.stream.flush()

    def update(self, n=1):
        self.current = min(self.total, self.current + n)
        self._draw()

    def _draw(self, force=False):
        if self.total <= 0:
            return

        fraction = self.current / self.total
        percent = int(fraction * 100)
        if not force and percent == self._last_percent:
            return

        self._last_percent = percent
        filled = int(self.width * fraction)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.time() - self.start_time if self.start_time else 0
        rate = self.current / elapsed if elapsed > 0 else 0

        message = (
            f"{self.label}: [{bar}] {self.current}/{self.total} "
            f"({percent:3d}%) {rate:.2f} it/s"
        )
        line_end = "" if self._is_tty else "\n"
        prefix = "\r" if self._is_tty else ""

        self.stream.write(prefix + message + line_end)
        self.stream.flush()


@contextlib.contextmanager
def joblib_progress(total, label="Progress", width=30):
    import joblib.parallel

    original_callback = joblib.parallel.BatchCompletionCallBack

    with ProgressBar(total, label=label, width=width) as progress:
        class ProgressCallback(original_callback):
            def __call__(self, *args, **kwargs):
                progress.update(n=self.batch_size)
                return super().__call__(*args, **kwargs)

        joblib.parallel.BatchCompletionCallBack = ProgressCallback
        try:
            yield progress
        finally:
            joblib.parallel.BatchCompletionCallBack = original_callback



def Gaussian(x, mu, sigma):

    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu)/ sigma)**2)
