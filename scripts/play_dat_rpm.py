import argparse  # noqa: INP001
import time
from collections import deque

import cv2
import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq

from evio.core.pacer import Pacer
from evio.source.dat_file import BatchRange, DatFileSource


class RPMDetector:
    """Detects RPM using FFT on accumulated event data."""
    
    def __init__(self, history_duration_s: float = 2.0, update_interval_s: float = 0.5):
        """
        Args:
            history_duration_s: How many seconds of history to keep for FFT
            update_interval_s: How often to update RPM estimate
        """
        self.history_duration_s = history_duration_s
        self.update_interval_s = update_interval_s
        
        # Store (timestamp_us, event_count) tuples
        self.event_history = deque()
        self.last_update_time = 0
        self.current_rpm = None
        self.rpm_confidence = 0.0
        
    def add_events(self, batch_range: BatchRange, num_events: int) -> None:
        """Add a batch of events to the history."""
        timestamp_us = batch_range.end_ts_us
        self.event_history.append((timestamp_us, num_events))
        
        # Remove old events outside history window
        cutoff_time = timestamp_us - (self.history_duration_s * 1e6)
        while self.event_history and self.event_history[0][0] < cutoff_time:
            self.event_history.popleft()
    
    def should_update(self, current_time_us: int) -> bool:
        """Check if enough time has passed to update RPM."""
        return (current_time_us - self.last_update_time) >= (self.update_interval_s * 1e6)
    
    def compute_rpm(self, current_time_us: int) -> tuple[float | None, float]:
        """
        Compute RPM using FFT on event rate signal.
        
        Returns:
            (rpm, confidence): RPM estimate and confidence score (0-1)
        """
        if len(self.event_history) < 20:
            return None, 0.0
        
        self.last_update_time = current_time_us
        
        # Convert history to arrays
        timestamps = np.array([t for t, _ in self.event_history])
        counts = np.array([c for _, c in self.event_history])
        
        # Create evenly sampled signal for FFT
        duration_s = (timestamps[-1] - timestamps[0]) / 1e6
        if duration_s < 0.5:
            return None, 0.0
        
        # Resample to uniform time grid
        sample_rate = 100  # Hz
        num_samples = int(duration_s * sample_rate)
        
        if num_samples < 50:
            return None, 0.0
        
        time_grid = np.linspace(timestamps[0], timestamps[-1], num_samples)
        
        # Interpolate event counts to uniform grid
        event_rate = np.interp(time_grid, timestamps, counts)
        
        # Apply window to reduce spectral leakage
        window = signal.windows.hann(len(event_rate))
        event_rate_windowed = event_rate * window
        
        # Compute FFT
        fft_result = rfft(event_rate_windowed)
        fft_freqs = rfftfreq(len(event_rate_windowed), 1/sample_rate)
        fft_magnitude = np.abs(fft_result)
        
        # Focus on reasonable RPM range (60-3000 RPM = 1-50 Hz)
        min_freq = 1.0  # 60 RPM
        max_freq = 50.0  # 3000 RPM
        
        valid_mask = (fft_freqs >= min_freq) & (fft_freqs <= max_freq)
        if not np.any(valid_mask):
            return None, 0.0
        
        valid_freqs = fft_freqs[valid_mask]
        valid_magnitudes = fft_magnitude[valid_mask]
        
        # Find peak frequency
        peak_idx = np.argmax(valid_magnitudes)
        peak_freq = valid_freqs[peak_idx]
        peak_magnitude = valid_magnitudes[peak_idx]
        
        # Calculate confidence based on peak prominence
        mean_magnitude = np.mean(valid_magnitudes)
        if mean_magnitude > 0:
            confidence = min(1.0, (peak_magnitude / mean_magnitude - 1) / 10)
        else:
            confidence = 0.0
        
        # Convert frequency to RPM
        rpm = peak_freq * 60
        
        # Only update if confidence is reasonable
        if confidence > 0.1:
            self.current_rpm = rpm
            self.rpm_confidence = confidence
        
        return self.current_rpm, self.rpm_confidence


def get_window(
    event_words: np.ndarray,
    time_order: np.ndarray,
    win_start: int,
    win_stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # get indexes corresponding to events within the window
    event_indexes = time_order[win_start:win_stop]
    words = event_words[event_indexes].astype(np.uint32, copy=False)
    x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
    y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    pixel_polarity = ((words >> 28) & 0xF) > 0

    return x_coords, y_coords, pixel_polarity


def get_frame(
    window: tuple[np.ndarray, np.ndarray, np.ndarray],
    width: int = 1280,
    height: int = 720,
    *,
    base_color: tuple[int, int, int] = (127, 127, 127),  # gray
    on_color: tuple[int, int, int] = (255, 255, 255),  # white
    off_color: tuple[int, int, int] = (0, 0, 0),  # black
) -> np.ndarray:
    x_coords, y_coords, polarities_on = window
    frame = np.full((height, width, 3), base_color, np.uint8)
    frame[y_coords[polarities_on], x_coords[polarities_on]] = on_color
    frame[y_coords[~polarities_on], x_coords[~polarities_on]] = off_color

    return frame


def draw_hud(
    frame: np.ndarray,
    pacer: Pacer,
    batch_range: BatchRange,
    rpm: float | None,
    confidence: float,
    *,
    color: tuple[int, int, int] = (0, 0, 0),  # black by default
) -> None:
    """Overlay timing info and RPM: wall time, recording time, playback speed, and RPM."""
    if pacer._t_start is None or pacer._e_start is None:
        return

    wall_time_s = time.perf_counter() - pacer._t_start
    rec_time_s = max(0.0, (batch_range.end_ts_us - pacer._e_start) / 1e6)

    if pacer.force_speed:
        first_row_str = (
            f"speed={pacer.speed:.2f}x"
            f"  drops/ms={pacer.instantaneous_drop_rate:.2f}"
            f"  avg(drops/ms)={pacer.average_drop_rate:.2f}"
        )
    else:
        first_row_str = (
            f"(target) speed={pacer.speed:.2f}x  force_speed = False, no drops"
        )

    second_row_str = f"wall={wall_time_s:7.3f}s  rec={rec_time_s:7.3f}s"

    # RPM display with confidence
    if rpm is not None:
        rpm_color = (0, 255, 0) if confidence > 0.5 else (0, 165, 255)  # Green if confident, orange if not
        rpm_str = f"RPM: {rpm:.1f} (conf: {confidence:.2f})"
    else:
        rpm_color = (0, 0, 255)  # Red
        rpm_str = "RPM: Calculating..."

    # first row
    cv2.putText(
        frame,
        first_row_str,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )

    # second row
    cv2.putText(
        frame,
        second_row_str,
        (8, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )

    # RPM row (larger and more prominent)
    cv2.putText(
        frame,
        rpm_str,
        (8, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        rpm_color,
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dat", help="Path to .dat file")
    parser.add_argument(
        "--window", type=float, default=10, help="Windows duration in ms"
    )
    parser.add_argument(
        "--speed", type=float, default=1, help="Playback speed (1 is real time)"
    )
    parser.add_argument(
        "--force-speed",
        action="store_true",
        help="Force the playback speed by dropping windows",
    )
    parser.add_argument(
        "--history", type=float, default=2.0, help="History duration for RPM calculation in seconds"
    )
    parser.add_argument(
        "--update-interval", type=float, default=0.5, help="RPM update interval in seconds"
    )
    args = parser.parse_args()

    src = DatFileSource(
        args.dat, width=1280, height=720, window_length_us=args.window * 1000
    )

    # Enforce playback speed via dropping:
    pacer = Pacer(speed=args.speed, force_speed=args.force_speed)
    
    # RPM detector
    rpm_detector = RPMDetector(
        history_duration_s=args.history,
        update_interval_s=args.update_interval
    )

    cv2.namedWindow("Evio Player", cv2.WINDOW_NORMAL)
    
    for batch_range in pacer.pace(src.ranges()):
        window = get_window(
            src.event_words,
            src.order,
            batch_range.start,
            batch_range.stop,
        )
        
        # Count events in this window
        num_events = batch_range.stop - batch_range.start
        rpm_detector.add_events(batch_range, num_events)
        
        # Update RPM estimate periodically
        if rpm_detector.should_update(batch_range.end_ts_us):
            rpm, confidence = rpm_detector.compute_rpm(batch_range.end_ts_us)
        else:
            rpm = rpm_detector.current_rpm
            confidence = rpm_detector.rpm_confidence
        
        frame = get_frame(window)
        draw_hud(frame, pacer, batch_range, rpm, confidence)

        cv2.imshow("Evio Player", frame)

        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break
            
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()