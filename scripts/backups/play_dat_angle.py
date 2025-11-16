import argparse  # noqa: INP001
import time
from collections import deque

import cv2
import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq

from evio.core.pacer import Pacer
from evio.source.dat_file import BatchRange, DatFileSource


class AngleEstimator:
    """Estimates viewing angle of rotating object based on event activity."""
    
    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.angle_history = deque(maxlen=20)
        self.baseline_area = None
        
    def estimate_from_events(
        self, 
        x_coords: np.ndarray, 
        y_coords: np.ndarray,
        num_events: int
    ) -> float:
        """
        Estimate viewing angle from event distribution.
        
        Returns angle in degrees (0° = face-on, 90° = edge-on)
        """
        if len(x_coords) < 10:
            return 0.0
        
        # Calculate spatial spread of events
        x_spread = np.std(x_coords) if len(x_coords) > 1 else 0
        y_spread = np.std(y_coords) if len(y_coords) > 1 else 0
        
        # Compute aspect ratio (how circular vs elongated)
        if y_spread > 1 and x_spread > 1:
            aspect_ratio = min(x_spread, y_spread) / max(x_spread, y_spread)
        else:
            aspect_ratio = 1.0
        
        # Aspect ratio of 1.0 = circular (face-on)
        # Aspect ratio approaching 0 = elongated (edge-on)
        # Convert to angle: circular distribution -> 0°, elongated -> 90°
        estimated_angle = (1.0 - aspect_ratio) * 90
        
        self.angle_history.append(estimated_angle)
        
        # Return smoothed angle
        return np.mean(list(self.angle_history))


class BladeDetector:
    """Detects number of blades using harmonic analysis of FFT peaks."""
    
    def __init__(self):
        self.detected_blades = None
        self.detection_confidence = 0.0
        self.blade_candidates = []
        
    def detect_blades(
        self,
        fft_freqs: np.ndarray,
        fft_magnitude: np.ndarray,
        min_rpm: float = 300,
        max_rpm: float = 3000
    ) -> tuple[int | None, float]:
        """
        Detect number of blades by analyzing FFT spectrum.
        
        Strategy: The dominant peak represents blade passage frequency.
        We test different blade counts to find which gives most consistent harmonics.
        
        Returns:
            (num_blades, confidence)
        """
        # Find peaks in spectrum
        mean_mag = np.mean(fft_magnitude)
        std_mag = np.std(fft_magnitude)
        threshold = mean_mag + 1.5 * std_mag
        
        peaks, properties = signal.find_peaks(
            fft_magnitude, 
            height=threshold,
            prominence=std_mag * 0.5,
            distance=3
        )
        
        if len(peaks) < 1:
            return None, 0.0
        
        peak_freqs = fft_freqs[peaks]
        peak_mags = properties['peak_heights']
        
        # Sort by magnitude
        sort_idx = np.argsort(peak_mags)[::-1]
        peak_freqs = peak_freqs[sort_idx]
        peak_mags = peak_mags[sort_idx]
        
        # The strongest peak is likely the blade passage frequency
        dominant_freq = peak_freqs[0]
        
        best_blade_count = None
        best_score = 0.0
        
        # Test blade counts from 2 to 8
        for test_blades in range(2, 9):
            # Calculate what RPM would be with this blade count
            fundamental_freq = dominant_freq / test_blades
            test_rpm = fundamental_freq * 60
            
            # Skip if RPM is unreasonable
            if test_rpm < min_rpm or test_rpm > max_rpm:
                continue
            
            # Score based on:
            # 1. How well harmonics align
            # 2. RPM being in sweet spot (900-1500 for typical fans)
            
            # Check for expected harmonics
            harmonics_found = 0
            expected_harmonics = [fundamental_freq * i * test_blades for i in range(1, 4)]
            
            for expected_freq in expected_harmonics:
                if expected_freq > fft_freqs[-1]:
                    break
                # Check if we have a peak near this frequency
                freq_diffs = np.abs(peak_freqs - expected_freq)
                if len(freq_diffs) > 0 and np.min(freq_diffs) < expected_freq * 0.15:
                    harmonics_found += 1
            
            # RPM preference (prefer 900-1500 range)
            rpm_score = 1.0
            if 900 <= test_rpm <= 1500:
                rpm_score = 1.5
            elif test_rpm < 600 or test_rpm > 2000:
                rpm_score = 0.7
            
            # Combined score
            harmonic_score = harmonics_found / 3.0
            total_score = harmonic_score * rpm_score
            
            if total_score > best_score:
                best_score = total_score
                best_blade_count = test_blades
        
        # Build consensus over time
        if best_blade_count is not None and best_score > 0.3:
            self.blade_candidates.append(best_blade_count)
            
            if len(self.blade_candidates) > 8:
                self.blade_candidates = self.blade_candidates[-15:]
                
                # Vote for most common
                unique, counts = np.unique(self.blade_candidates, return_counts=True)
                most_common_idx = np.argmax(counts)
                most_common = unique[most_common_idx]
                consistency = counts[most_common_idx] / len(self.blade_candidates)
                
                if consistency > 0.5:
                    self.detected_blades = int(most_common)
                    self.detection_confidence = consistency
                    return self.detected_blades, self.detection_confidence
        
        return self.detected_blades, self.detection_confidence if self.detected_blades else 0.0


class RPMDetector:
    """Detects RPM using FFT with automatic blade detection and angle correction."""
    
    def __init__(
        self, 
        history_duration_s: float = 2.0, 
        update_interval_s: float = 0.5,
        num_blades: int | None = None
    ):
        self.history_duration_s = history_duration_s
        self.update_interval_s = update_interval_s
        self.num_blades = num_blades
        self.auto_detect_blades = (num_blades is None)
        
        self.event_history = deque()
        self.last_update_time = 0
        self.current_rpm = None
        self.apparent_rpm = None
        self.rpm_confidence = 0.0
        self.rpm_history = deque(maxlen=10)
        
        self.blade_detector = BladeDetector()
        
        # Store raw FFT data for debugging
        self.last_fft_freqs = None
        self.last_fft_mags = None
        
    def add_events(self, batch_range: BatchRange, num_events: int) -> None:
        """Add a batch of events to the history."""
        timestamp_us = batch_range.end_ts_us
        self.event_history.append((timestamp_us, num_events))
        
        cutoff_time = timestamp_us - (self.history_duration_s * 1e6)
        while self.event_history and self.event_history[0][0] < cutoff_time:
            self.event_history.popleft()
    
    def should_update(self, current_time_us: int) -> bool:
        """Check if enough time has passed to update RPM."""
        return (current_time_us - self.last_update_time) >= (self.update_interval_s * 1e6)
    
    def compute_rpm(
        self, 
        current_time_us: int,
        viewing_angle_deg: float = 0.0
    ) -> tuple[float | None, float | None, float]:
        """
        Compute RPM using FFT with angle correction and blade detection.
        
        Returns:
            (corrected_rpm, apparent_rpm, confidence)
        """
        if len(self.event_history) < 20:
            return None, None, 0.0
        
        self.last_update_time = current_time_us
        
        timestamps = np.array([t for t, _ in self.event_history])
        counts = np.array([c for _, c in self.event_history])
        
        duration_s = (timestamps[-1] - timestamps[0]) / 1e6
        if duration_s < 0.5:
            return None, None, 0.0
        
        sample_rate = 100  # Hz
        num_samples = int(duration_s * sample_rate)
        
        if num_samples < 50:
            return None, None, 0.0
        
        time_grid = np.linspace(timestamps[0], timestamps[-1], num_samples)
        event_rate = np.interp(time_grid, timestamps, counts)
        
        # Remove DC component and trends
        event_rate_detrended = signal.detrend(event_rate)
        
        # Apply window
        window = signal.windows.hann(len(event_rate_detrended))
        event_rate_windowed = event_rate_detrended * window
        
        # Compute FFT
        fft_result = rfft(event_rate_windowed)
        fft_freqs = rfftfreq(len(event_rate_windowed), 1/sample_rate)
        fft_magnitude = np.abs(fft_result)
        
        # Focus on reasonable frequency range
        min_freq = 5.0   # 300 RPM / 60 = 5 Hz minimum fundamental
        max_freq = 50.0  # 3000 RPM / 60 = 50 Hz maximum fundamental
        
        valid_mask = (fft_freqs >= min_freq) & (fft_freqs <= max_freq)
        if not np.any(valid_mask):
            return None, None, 0.0
        
        valid_freqs = fft_freqs[valid_mask]
        valid_magnitudes = fft_magnitude[valid_mask]
        
        # Store for debugging
        self.last_fft_freqs = valid_freqs
        self.last_fft_mags = valid_magnitudes
        
        # Auto-detect blade count if needed
        if self.auto_detect_blades:
            detected_blades, blade_confidence = self.blade_detector.detect_blades(
                valid_freqs, valid_magnitudes
            )
            if detected_blades is not None and blade_confidence > 0.4:
                self.num_blades = detected_blades
        
        # Default to 3 blades if still unknown
        if self.num_blades is None:
            self.num_blades = 3
        
        # Find dominant peak
        peak_idx = np.argmax(valid_magnitudes)
        peak_freq = valid_freqs[peak_idx]
        peak_magnitude = valid_magnitudes[peak_idx]
        
        # Calculate confidence using SNR
        mean_magnitude = np.mean(valid_magnitudes)
        std_magnitude = np.std(valid_magnitudes)
        
        if std_magnitude > 0:
            snr = (peak_magnitude - mean_magnitude) / std_magnitude
            confidence = min(1.0, snr / 5.0)
        else:
            confidence = 0.0
        
        # CRITICAL: Determine if peak is fundamental or harmonic
        # For a 3-blade fan at 1100 RPM:
        # - Fundamental: 18.3 Hz (rotation rate)
        # - 1st harmonic: 55 Hz (blade passage = 3 × 18.3)
        # - 2nd harmonic: 110 Hz (2 × blade passage)
        
        # The dominant peak is usually the blade passage frequency (1st harmonic)
        # So we divide by num_blades to get rotation rate
        rotation_freq_from_peak = peak_freq / self.num_blades
        apparent_rpm_option1 = rotation_freq_from_peak * 60
        
        # BUT: Sometimes the peak is actually the fundamental (rotation rate itself)
        # This happens with strong motion blur or low blade count
        apparent_rpm_option2 = peak_freq * 60
        
        # Heuristic: blade passage frequency is usually stronger for fans
        # Check if there's a peak near peak_freq / num_blades (the fundamental)
        fundamental_freq_candidate = peak_freq / self.num_blades
        
        # Look for sub-harmonics to determine which interpretation is correct
        subharmonic_present = False
        if len(valid_freqs) > 1:
            freq_tolerance = fundamental_freq_candidate * 0.2
            distances = np.abs(valid_freqs - fundamental_freq_candidate)
            if np.min(distances) < freq_tolerance:
                # Found a peak near the fundamental - our peak is likely 1st harmonic
                subharmonic_present = True
        
        # Choose interpretation
        if subharmonic_present:
            # Peak is blade passage frequency
            apparent_rpm = apparent_rpm_option1
        else:
            # Peak might be fundamental - check which gives more reasonable RPM
            if 800 <= apparent_rpm_option1 <= 1500:
                apparent_rpm = apparent_rpm_option1
            elif 800 <= apparent_rpm_option2 <= 1500:
                apparent_rpm = apparent_rpm_option2
            else:
                # Default to blade passage interpretation
                apparent_rpm = apparent_rpm_option1
        
        # Angle correction: apparent_velocity = true_velocity * cos(angle)
        # So: true_rpm = apparent_rpm / cos(angle)
        angle_rad = np.radians(viewing_angle_deg)
        cos_angle = np.cos(angle_rad)
        
        if cos_angle > 0.2:  # Avoid extreme corrections
            corrected_rpm = apparent_rpm / cos_angle
        else:
            corrected_rpm = apparent_rpm
            confidence *= 0.3
        
        # Update if confidence is reasonable
        if confidence > 0.15:
            self.current_rpm = corrected_rpm
            self.apparent_rpm = apparent_rpm
            self.rpm_confidence = confidence
            self.rpm_history.append(corrected_rpm)
        
        return self.current_rpm, self.apparent_rpm, self.rpm_confidence
    
    def get_rpm_trend(self) -> str:
        """Get trend indicator for varying RPM."""
        if len(self.rpm_history) < 3:
            return ""
        
        recent = list(self.rpm_history)[-3:]
        if recent[-1] > recent[0] + 50:
            return " ↑"
        elif recent[-1] < recent[0] - 50:
            return " ↓"
        else:
            return " →"


def get_window(
    event_words: np.ndarray,
    time_order: np.ndarray,
    win_start: int,
    win_stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    base_color: tuple[int, int, int] = (127, 127, 127),
    on_color: tuple[int, int, int] = (255, 255, 255),
    off_color: tuple[int, int, int] = (0, 0, 0),
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
    corrected_rpm: float | None,
    apparent_rpm: float | None,
    confidence: float,
    viewing_angle: float,
    rpm_detector: RPMDetector,
    *,
    color: tuple[int, int, int] = (0, 0, 0),
) -> None:
    """Overlay timing info, blade count, angle, and RPM data."""
    if pacer._t_start is None or pacer._e_start is None:
        return

    wall_time_s = time.perf_counter() - pacer._t_start
    rec_time_s = max(0.0, (batch_range.end_ts_us - pacer._e_start) / 1e6)

    if pacer.force_speed:
        first_row = f"speed={pacer.speed:.2f}x  drops/ms={pacer.instantaneous_drop_rate:.2f}"
    else:
        first_row = f"speed={pacer.speed:.2f}x (no drops)"

    second_row = f"wall={wall_time_s:.2f}s  rec={rec_time_s:.2f}s"

    # Blade count display
    if rpm_detector.auto_detect_blades:
        blade_conf = rpm_detector.blade_detector.detection_confidence
        blade_color = (0, 255, 0) if blade_conf > 0.5 else (255, 165, 0)
        if rpm_detector.num_blades:
            blade_str = f"Blades: {rpm_detector.num_blades} (auto, conf: {blade_conf:.2f})"
        else:
            blade_str = "Blades: Detecting..."
    else:
        blade_color = (200, 200, 200)
        blade_str = f"Blades: {rpm_detector.num_blades} (fixed)"

    # Angle display
    angle_str = f"Angle: {viewing_angle:.1f}°"
    angle_color = (100, 200, 100) if viewing_angle < 30 else (255, 165, 0)
    
    # RPM display
    if corrected_rpm is not None and apparent_rpm is not None:
        rpm_color = (0, 255, 0) if confidence > 0.5 else (0, 165, 255)
        trend = rpm_detector.get_rpm_trend()
        rpm_str = f"True RPM: {corrected_rpm:.1f}{trend}"
        apparent_str = f"Apparent: {apparent_rpm:.1f}  Conf: {confidence:.2f}"
    else:
        rpm_color = (0, 0, 255)
        rpm_str = "RPM: Calculating..."
        apparent_str = ""

    # Draw text
    y_pos = 20
    for text in [first_row, second_row]:
        cv2.putText(frame, text, (8, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                   0.45, color, 1, cv2.LINE_AA)
        y_pos += 18

    # Blade count
    cv2.putText(frame, blade_str, (8, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
               0.6, blade_color, 2, cv2.LINE_AA)
    y_pos += 25

    # Angle
    cv2.putText(frame, angle_str, (8, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
               0.6, angle_color, 2, cv2.LINE_AA)
    y_pos += 25

    # RPM (most prominent)
    cv2.putText(frame, rpm_str, (8, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
               0.75, rpm_color, 2, cv2.LINE_AA)
    y_pos += 25
    
    if apparent_str:
        cv2.putText(frame, apparent_str, (8, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (200, 200, 200), 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dat", help="Path to .dat file")
    parser.add_argument("--window", type=float, default=10, 
                       help="Window duration in ms")
    parser.add_argument("--speed", type=float, default=1, 
                       help="Playback speed")
    parser.add_argument("--force-speed", action="store_true",
                       help="Force playback speed")
    parser.add_argument("--history", type=float, default=2.5,
                       help="History duration (s)")
    parser.add_argument("--update-interval", type=float, default=0.3,
                       help="RPM update interval (s)")
    parser.add_argument("--blades", type=int, default=None,
                       help="Number of blades (auto-detect if not specified)")
    parser.add_argument("--fixed-angle", type=float, default=None,
                       help="Fixed viewing angle (overrides auto-detection)")
    parser.add_argument("--disable-angle-correction", action="store_true",
                       help="Disable angle correction (useful for debugging)")
    parser.add_argument("--debug", action="store_true",
                       help="Print FFT debug info to console")
    args = parser.parse_args()

    src = DatFileSource(
        args.dat, width=1280, height=720, window_length_us=args.window * 1000
    )

    pacer = Pacer(speed=args.speed, force_speed=args.force_speed)
    
    rpm_detector = RPMDetector(
        history_duration_s=args.history,
        update_interval_s=args.update_interval,
        num_blades=args.blades
    )
    
    angle_estimator = AngleEstimator(width=1280, height=720)

    cv2.namedWindow("Evio Player", cv2.WINDOW_NORMAL)
    
    print("Starting playback...")
    if args.blades is None:
        print("Auto-detecting number of blades...")
    else:
        print(f"Using {args.blades} blades")
    
    frame_count = 0
    for batch_range in pacer.pace(src.ranges()):
        window = get_window(
            src.event_words,
            src.order,
            batch_range.start,
            batch_range.stop,
        )
        
        x_coords, y_coords, polarities = window
        num_events = batch_range.stop - batch_range.start
        
        # Estimate or use fixed angle
        if args.fixed_angle is not None:
            viewing_angle = args.fixed_angle
        elif args.disable_angle_correction:
            viewing_angle = 0.0
        else:
            viewing_angle = angle_estimator.estimate_from_events(
                x_coords, y_coords, num_events
            )
        
        rpm_detector.add_events(batch_range, num_events)
        
        if rpm_detector.should_update(batch_range.end_ts_us):
            corrected_rpm, apparent_rpm, confidence = rpm_detector.compute_rpm(
                batch_range.end_ts_us, viewing_angle
            )
            
            # Debug output
            if args.debug and frame_count % 10 == 0 and rpm_detector.last_fft_freqs is not None:
                print(f"\n=== FFT Debug (frame {frame_count}) ===")
                freqs = rpm_detector.last_fft_freqs
                mags = rpm_detector.last_fft_mags
                
                # Find top 5 peaks
                top_indices = np.argsort(mags)[-5:][::-1]
                print("Top 5 frequency peaks:")
                for i, idx in enumerate(top_indices):
                    freq_hz = freqs[idx]
                    rpm_if_fundamental = freq_hz * 60
                    rpm_if_3blade_harmonic = (freq_hz / 3) * 60
                    print(f"  {i+1}. {freq_hz:.2f} Hz (mag: {mags[idx]:.1f})")
                    print(f"     -> If fundamental: {rpm_if_fundamental:.0f} RPM")
                    print(f"     -> If 3-blade harmonic: {rpm_if_3blade_harmonic:.0f} RPM")
                
                print(f"Detected: {rpm_detector.num_blades} blades, {corrected_rpm:.1f} RPM")
                print("=" * 40)
        else:
            corrected_rpm = rpm_detector.current_rpm
            apparent_rpm = rpm_detector.apparent_rpm
            confidence = rpm_detector.rpm_confidence
        
        frame_count += 1
        frame = get_frame(window)
        draw_hud(frame, pacer, batch_range, corrected_rpm, apparent_rpm, 
                confidence, viewing_angle, rpm_detector)

        cv2.imshow("Evio Player", frame)

        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break
    
    # Print final results
    if rpm_detector.num_blades:
        print(f"\nDetected {rpm_detector.num_blades} blades")
    if corrected_rpm:
        print(f"Final RPM: {corrected_rpm:.1f}")
            
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()