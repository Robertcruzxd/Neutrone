import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from evio.core.pacer import Pacer
from evio.source.dat_file import BatchRange, DatFileSource


class DatasetGenerator:
    """Generate labeled dataset from event camera recordings."""
    
    def __init__(
        self,
        output_dir: str,
        delta_ms: float = 100,
        width: int = 1280,
        height: int = 720
    ):
        self.output_dir = Path(output_dir)
        self.delta_us = int(delta_ms * 1000)  # Convert to microseconds
        self.width = width
        self.height = height
        
        # Create directory structure
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata = {
            "delta_ms": delta_ms,
            "width": width,
            "height": height,
            "frames": []
        }
        
        self.next_capture_time = 0
        self.frame_counter = 0
        
        # Track drone position across frames for stability and motion prediction
        self.last_drone_center = None
        self.last_propeller_positions = None
        self.drone_velocity = None  # Track movement direction
        self.tracking_confidence = 0  # How confident we are in current tracking
        self.last_size = None  # Track object size for consistency (min_spread, max_spread)
        
    def _find_all_propeller_candidates(self, x_coords, y_coords):
        """
        Find ALL potential propeller clusters in the frame.
        Propellers = VERY tight, VERY dense black pixel clusters (THE key signature).
        Returns list of (x, y, density, tightness, event_count) sorted by quality.
        """
        if len(x_coords) < 30:
            return []
        
        # FINE grid to catch all propellers (even overlapping ones)
        grid_size = 8  # Very fine for maximum resolution
        x_bins = np.arange(200, self.width, grid_size)
        y_bins = np.arange(0, self.height + grid_size, grid_size)
        
        activity_map, x_edges, y_edges = np.histogram2d(
            x_coords, y_coords,
            bins=(x_bins, y_bins)
        )
        
        # Find ALL significant peaks
        flat_activity = activity_map.flatten()
        threshold = max(5, np.percentile(flat_activity, 80))  # Top 20% activity
        significant = np.where(flat_activity >= threshold)[0]
        
        raw_candidates = []
        for idx in significant:
            i, j = np.unravel_index(idx, activity_map.shape)
            if i >= len(x_bins) - 1 or j >= len(y_bins) - 1:
                continue
            
            prop_x = (x_bins[i] + x_bins[i+1]) / 2
            prop_y = (y_bins[j] + y_bins[j+1]) / 2
            
            # PROPELLER SIGNATURE: Ultra-dense, ultra-tight black pixels
            # Key: Propellers have 10-25 events within 8px (NOT 50+ like merged blob)
            dist_from_prop = np.sqrt((x_coords - prop_x)**2 + (y_coords - prop_y)**2)
            within_5px = np.sum(dist_from_prop < 5)   # Ultra core
            within_8px = np.sum(dist_from_prop < 8)   # Tight core  
            within_12px = np.sum(dist_from_prop < 12)  # Core
            within_18px = np.sum(dist_from_prop < 18)  # Extended
            
            # Single propeller: 8-30 events in core, not 50+
            # Merged blob: 50-80 events in same area
            if within_8px < 6 or within_12px < 10:
                continue
            
            # Multi-level tightness
            ultra_tight = within_5px / (within_8px + 1)
            core_tight = within_8px / (within_12px + 1) 
            falloff = within_12px / (within_18px + 1)
            
            # Propeller has sharp falloff (tight core, sparse edges)
            if (ultra_tight < 0.25 and core_tight < 0.3 and falloff < 0.35):
                continue
            
            # Quality score for ranking
            combined_tightness = (ultra_tight * 0.4 + core_tight * 0.35 + falloff * 0.25)
            quality = combined_tightness * within_12px
            
            raw_candidates.append({
                'pos': (prop_x, prop_y),
                'quality': quality,
                'tightness': combined_tightness,
                'count': within_12px,
                'core_count': within_8px
            })
        
        # Non-maximum suppression: keep best propellers, remove duplicates
        raw_candidates.sort(key=lambda c: c['quality'], reverse=True)
        
        propeller_candidates = []
        MIN_SEP = 15  # Minimum distance between propeller centers
        
        for cand in raw_candidates:
            px, py = cand['pos']
            
            # Check if too close to already selected
            too_close = False
            for selected_x, selected_y, _, _, _ in propeller_candidates:
                if np.sqrt((px - selected_x)**2 + (py - selected_y)**2) < MIN_SEP:
                    too_close = True
                    break
            
            if not too_close:
                propeller_candidates.append((
                    px, py,
                    cand['tightness'],
                    cand['tightness'],  # Use same for compatibility
                    cand['count']
                ))
                
                # Limit to 6 candidates max
                if len(propeller_candidates) >= 6:
                    break
        
        return propeller_candidates
    
    def _find_propeller_clusters(self, x_coords, y_coords, drone_center):
        """Find propeller positions - they appear as tight black pixel clusters."""
        cx, cy = drone_center
        
        # Look within 100px of drone center (tighter focus)
        distances = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
        drone_mask = distances < 100
        
        if np.sum(drone_mask) < 30:
            return []
        
        drone_x = x_coords[drone_mask]
        drone_y = y_coords[drone_mask]
        
        # Fine grid (10px) to find propeller peaks
        grid_size = 10
        activity_map, x_bins, y_bins = np.histogram2d(
            drone_x, drone_y,
            bins=(200 // grid_size, 200 // grid_size),
            range=[[cx - 100, cx + 100], [cy - 100, cy + 100]]
        )
        
        # Find top activity peaks
        flat_activity = activity_map.flatten()
        top_indices = np.argsort(flat_activity)[-20:][::-1]
        
        propeller_candidates = []
        for idx in top_indices:
            i, j = np.unravel_index(idx, activity_map.shape)
            prop_x = x_bins[i] + grid_size / 2
            prop_y = y_bins[j] + grid_size / 2
            activity = activity_map[i, j]
            
            if activity < 5:
                continue
            
            # Verify tight clustering (propeller signature)
            dist_from_prop = np.sqrt((drone_x - prop_x)**2 + (drone_y - prop_y)**2)
            very_tight = np.sum(dist_from_prop < 12)  # Very tight
            tight = np.sum(dist_from_prop < 20)       # Tight
            
            if very_tight < 3 or tight < 8:
                continue
            
            prop_density = very_tight / (tight + 1)
            propeller_candidates.append((prop_x, prop_y, prop_density, activity))
        
        # Sort by density then activity
        propeller_candidates.sort(key=lambda p: (p[2], p[3]), reverse=True)
        
        # Select up to 4 with minimum 25px separation
        propellers = []
        for prop_x, prop_y, density, activity in propeller_candidates:
            too_close = any(np.sqrt((prop_x - px)**2 + (prop_y - py)**2) < 25 
                          for px, py in propellers)
            
            if not too_close:
                propellers.append((prop_x, prop_y))
                if len(propellers) >= 4:
                    break
        
        return propellers
    
    def _detect_drone_region(
        self, 
        x_coords: np.ndarray, 
        y_coords: np.ndarray,
        grid_size: int = 50
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        PROPELLER-FIRST DETECTION: Find dense propeller clusters, then infer drone position.
        Propellers are THE distinguishing feature - tight, dense black pixel blobs.
        """
        if len(x_coords) < 10:
            return x_coords, y_coords, {}
        
        # PRE-FILTER: Remove noise zone (x < 200) unless we're confidently tracking there
        # This prevents the noise from contaminating all subsequent calculations
        if not (self.last_drone_center and self.last_drone_center[0] < 250 and self.tracking_confidence > 0.6):
            clean_mask = x_coords >= 200
            x_coords = x_coords[clean_mask]
            y_coords = y_coords[clean_mask]
            if len(x_coords) < 10:
                return x_coords, y_coords, {}
        
        # STEP 1: Find ALL propeller candidates (tight dense clusters)
        propeller_candidates = self._find_all_propeller_candidates(x_coords, y_coords)
        
        if len(propeller_candidates) == 0:
            # Fallback to old method
            return self._detect_drone_region_fallback(x_coords, y_coords, grid_size)
        
        # STEP 2: Group propellers into drone candidates
        # Try each propeller as potential drone center
        best_drone = None
        best_score = 0
        
        # Look for multiple separated propellers
        for i, (px, py, density, tightness, count) in enumerate(propeller_candidates):
            # SPECIAL CASE: Single super-dense cluster (merged propellers from head-on angle)
            # When all propellers merge, we get ONE blob with 40-80 events
            # This is STRONGER evidence than multiple weak clusters (airplane)
            if count >= 50 and tightness > 0.45:  # Super-dense single blob
                # Treat this as the drone (no nearby propellers needed)
                drone_cx, drone_cy = px, py
                
                # Estimate size from event spread around this point
                dist_from_center = np.sqrt((x_coords - px)**2 + (y_coords - py)**2)
                nearby_mask = dist_from_center < 50
                if np.sum(nearby_mask) > 40:
                    nearby_x = x_coords[nearby_mask]
                    nearby_y = y_coords[nearby_mask]
                    x_spread = np.std(nearby_x)
                    y_spread = np.std(nearby_y)
                    
                    # Merged propellers: compact 20-60px spread
                    if 12 < x_spread < 70 and 8 < y_spread < 60:
                        # EMERGENCY FALLBACK SCORE: Should LOSE to 2+ separated propellers
                        # Multi-propeller detection is FAR better evidence
                        size_score = (x_spread * y_spread) / 100
                        score = count * 30 * tightness * 5 + density * 300 + size_score * 50
                        
                        # Tracking bonus (minimal)
                        if self.last_drone_center and self.tracking_confidence > 0.5:
                            pred_dist = np.sqrt((drone_cx - self.last_drone_center[0])**2 + 
                                               (drone_cy - self.last_drone_center[1])**2)
                            if pred_dist < 150:
                                score *= (1 + 2.0 / (pred_dist / 20 + 1))  # Weak tracking bonus
                        
                        if score > best_score:
                            best_score = score
                            best_drone = {
                                'center': (drone_cx, drone_cy),
                                'propellers': [(px, py)],
                                'num_props': 1,  # Merged
                                'spread': (x_spread, y_spread),
                                'tightness': tightness
                            }
            
            # NORMAL CASE: Find nearby propellers (within 120px)
            # IMPROVED: Handle close propellers (intersecting circles) by allowing closer spacing
            nearby_props = []
            for j, (px2, py2, d2, t2, c2) in enumerate(propeller_candidates):
                if i == j:
                    continue
                dist = np.sqrt((px - px2)**2 + (py - py2)**2)
                # RELAXED: Allow propellers as close as 20px (overlapping case)
                # But not TOO far (120px = typical drone extent)
                if 20 < dist < 120:
                    nearby_props.append((px2, py2, d2, t2, c2, dist))
            
            # Need at least 1 nearby propeller (2 total minimum for separated case)
            # But LIMIT to reasonable drone size (4-6 propellers max)
            if len(nearby_props) < 1:
                continue
            
            # Sort nearby propellers by quality (distance weighted)
            nearby_props.sort(key=lambda p: p[3] * p[4] / (p[5] + 10), reverse=True)
            
            # LIMIT: Take best 5 nearby propellers max (total 6 with center)
            # A drone has 4 propellers, so 4-6 detected is reasonable
            nearby_props = nearby_props[:5]
            
            # Calculate drone center from propeller group
            all_props_x = [px] + [p[0] for p in nearby_props]
            all_props_y = [py] + [p[1] for p in nearby_props]
            drone_cx = np.mean(all_props_x)
            drone_cy = np.mean(all_props_y)
            
            # Calculate number of propellers first
            num_propellers = len(nearby_props) + 1
            
            # Calculate spread (drone size)
            x_spread = np.std(all_props_x) if len(all_props_x) > 1 else 0
            y_spread = np.std(all_props_y) if len(all_props_y) > 1 else 0
            
            # Validate drone size (not too small = distant object, not too large = noise)
            # For 2 propellers: allow very tight (overlapping case)
            # For 3+ propellers: require reasonable spread
            if num_propellers == 2:
                # Two propellers: can be close (overlapping case)
                if x_spread < 5 or x_spread > 80 or y_spread > 50:
                    continue
            else:
                # 3+ propellers: need decent spread
                if x_spread < 8 or x_spread > 80 or y_spread < 2 or y_spread > 50:
                    continue
            
            # Score based on: number of propellers, density, tightness, size consistency
            avg_density = (density + sum(p[2] for p in nearby_props)) / num_propellers
            avg_tightness = (tightness + sum(p[3] for p in nearby_props)) / num_propellers
            total_events = count + sum(p[4] for p in nearby_props)  # Total events in all propellers
            size_score = (x_spread * y_spread) / 100  # Prefer larger (closer) objects
            
            # SEPARATED PROPELLERS SCORE: Multiple clusters - DOMINANT evidence type
            # This should STRONGLY beat merged detection when 3-4 propellers found
            propeller_bonus = 1.0
            if num_propellers >= 4:
                propeller_bonus = 20.0  # 4+ propellers = MASSIVE boost (ideal, undeniable drone)
            elif num_propellers == 3:
                propeller_bonus = 12.0  # 3 propellers = huge boost (clear drone)
            elif num_propellers == 2:
                propeller_bonus = 6.0   # 2 propellers = strong boost (likely drone)
            
            score = (num_propellers * 3500 * propeller_bonus + 
                    total_events * 50 + 
                    avg_tightness * 1000 + 
                    avg_density * 800 + 
                    size_score * 400)
            
            # Bonus if near predicted position
            if self.last_drone_center and self.tracking_confidence > 0.5:
                pred_dist = np.sqrt((drone_cx - self.last_drone_center[0])**2 + 
                                   (drone_cy - self.last_drone_center[1])**2)
                if pred_dist < 150:
                    score *= (1 + 5.0 / (pred_dist / 20 + 1))  # Strong proximity bonus
            
            if score > best_score:
                best_score = score
                best_drone = {
                    'center': (drone_cx, drone_cy),
                    'propellers': [(px, py)] + [(p[0], p[1]) for p in nearby_props],
                    'num_props': num_propellers,
                    'spread': (x_spread, y_spread),
                    'tightness': avg_tightness
                }
        
        if not best_drone:
            # No good drone candidate found - drone might be out of frame
            return self._detect_drone_region_fallback(x_coords, y_coords, grid_size)
        
        # STEP 3: Validate detection is reasonable (not spurious noise)
        # If score is too low, drone might actually be out of frame
        # LOWERED after scoring adjustments (merged=~2000-5000, multi-prop=~8000-30000)
        MIN_CONFIDENCE_SCORE = 1500  # Below this, likely not the actual drone
        if best_score < MIN_CONFIDENCE_SCORE:
            # Check if we have strong tracking history - if so, drone likely left frame
            if self.last_drone_center and self.tracking_confidence > 0.6:
                # Drone was being tracked well but now nothing good detected
                # Likely out of frame - decay confidence and use fallback
                self.tracking_confidence *= 0.5
                return self._detect_drone_region_fallback(x_coords, y_coords, grid_size)
        
        # STEP 4: Extract events around detected drone with TIGHTER bounding box
        drone_cx, drone_cy = best_drone['center']
        
        # IMPROVED: Dynamic radius based on actual propeller positions
        propeller_positions = best_drone['propellers']
        if len(propeller_positions) >= 2:
            # Calculate actual extent of propellers
            prop_xs = [p[0] for p in propeller_positions]
            prop_ys = [p[1] for p in propeller_positions]
            max_prop_dist = max(
                np.sqrt((px - drone_cx)**2 + (py - drone_cy)**2)
                for px, py in propeller_positions
            )
            # Add 30px margin beyond furthest propeller
            radius = int(max_prop_dist + 30)
            # Clamp to reasonable bounds
            radius = max(50, min(100, radius))
        else:
            # Single merged propeller case - use spread-based radius
            x_spread, y_spread = best_drone['spread']
            radius = int(max(x_spread, y_spread) * 2.5)
            radius = max(40, min(80, radius))
        
        distances = np.sqrt((x_coords - drone_cx)**2 + (y_coords - drone_cy)**2)
        drone_mask = distances < radius
        
        drone_x = x_coords[drone_mask]
        drone_y = y_coords[drone_mask]
        
        # Update tracking state
        self.last_drone_center = (drone_cx, drone_cy)
        self.last_size = best_drone['spread']
        
        # Update confidence based on propeller detection quality
        if best_drone['num_props'] >= 3 and best_drone['tightness'] > 0.4:
            self.tracking_confidence = min(1.0, self.tracking_confidence + 0.3)
        else:
            self.tracking_confidence = max(0.5, self.tracking_confidence)
        
        # Update velocity
        if hasattr(self, '_last_pos') and self._last_pos:
            vx = drone_cx - self._last_pos[0]
            vy = drone_cy - self._last_pos[1]
            if self.drone_velocity:
                prev_vx, prev_vy = self.drone_velocity
                vx = 0.7 * vx + 0.3 * prev_vx
                vy = 0.7 * vy + 0.3 * prev_vy
            self.drone_velocity = (vx, vy)
        self._last_pos = (drone_cx, drone_cy)
        
        return drone_x, drone_y, {
            'bbox': (
                int(drone_cx - radius),
                int(drone_cy - radius),
                int(drone_cx + radius),
                int(drone_cy + radius)
            ),
            'propellers': best_drone['propellers'],
            'num_propellers': best_drone['num_props'],
            'tracking_confidence': self.tracking_confidence
        }
    
    def _detect_drone_region_fallback(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        grid_size: int = 50
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        Fallback detection when propeller-first approach doesn't find enough candidates.
        Use velocity prediction if available, or mark as out-of-frame.
        """
        if len(x_coords) < 10:
            # Very few events - drone definitely out of frame
            self.tracking_confidence = 0.0
            return np.array([]), np.array([]), {'out_of_frame': True}
        
        # Use velocity prediction if we have tracking history
        if self.last_drone_center and self.drone_velocity and self.tracking_confidence > 0.3:
            vx, vy = self.drone_velocity
            drone_cx = self.last_drone_center[0] + vx
            drone_cy = self.last_drone_center[1] + vy
            
            # Check if predicted position is outside frame bounds
            if drone_cx < 0 or drone_cx > self.width or drone_cy < 0 or drone_cy > self.height:
                # Drone predicted to be out of frame
                self.tracking_confidence = 0.0
                return np.array([]), np.array([]), {'out_of_frame': True}
            
            self.tracking_confidence *= 0.7  # Decay confidence when using prediction
        else:
            # No tracking history - check if there's ANY reasonable activity
            # If all events are just scattered noise, drone is likely out
            x_spread = np.std(x_coords)
            y_spread = np.std(y_coords)
            
            # If events are too scattered (>300px) or too few, likely no drone
            if x_spread > 300 or y_spread > 300 or len(x_coords) < 100:
                self.tracking_confidence = 0.0
                return np.array([]), np.array([]), {'out_of_frame': True}
            
            # Use median of clean events as last resort
            drone_cx = np.median(x_coords)
            drone_cy = np.median(y_coords)
            self.tracking_confidence = 0.1
        
        # Extract events around predicted position
        radius = 100
        distances = np.sqrt((x_coords - drone_cx)**2 + (y_coords - drone_cy)**2)
        drone_mask = distances < radius
        
        drone_x = x_coords[drone_mask]
        drone_y = y_coords[drone_mask]
        
        # If very few events in predicted area, drone likely out of frame
        if len(drone_x) < 50:
            self.tracking_confidence = 0.0
            return np.array([]), np.array([]), {'out_of_frame': True}
        
        self.last_drone_center = (drone_cx, drone_cy)
        
        return drone_x, drone_y, {
            'bbox': (
                int(drone_cx - radius),
                int(drone_cy - radius),
                int(drone_cx + radius),
                int(drone_cy + radius)
            ),
            'tracking_confidence': self.tracking_confidence,
            'fallback': True
        }
        
        # Create activity heatmap
        x_bins = np.arange(0, self.width + grid_size, grid_size)
        y_bins = np.arange(0, self.height + grid_size, grid_size)
        activity_map, _, _ = np.histogram2d(x_coords, y_coords, bins=[x_bins, y_bins])
        
        # Find best candidate
        best_score = -1
        best_center = None
        
        flat_activity = activity_map.flatten()
        top_indices = np.argsort(flat_activity)[-25:][::-1]
        
        # Progressive search restriction based on tracking confidence
        # Once we're tracking well, don't jump to distant regions
        search_radius = None
        if predicted_center:
            if self.tracking_confidence > 0.9:
                search_radius = 80   # Very tight when locked
            elif self.tracking_confidence > 0.7:
                search_radius = 120  # Tight when confident
            elif self.tracking_confidence > 0.5:
                search_radius = 180  # Medium restriction
        
        for idx in top_indices:
            i, j = np.unravel_index(idx, activity_map.shape)
            cx = x_bins[i] + grid_size / 2
            cy = y_bins[j] + grid_size / 2
            activity = activity_map[i, j]
            
            # Progressive radius restriction based on confidence
            if search_radius:
                dist_from_pred = np.sqrt((cx - predicted_center[0])**2 + (cy - predicted_center[1])**2)
                if dist_from_pred > search_radius:
                    continue
            
            # HARD FILTER: Skip leftmost noise zone (x < 200) unless confidently tracking there
            # This region has dense sensor noise (~90,000 events) that confuses detection
            if cx < 200:
                # Only allow if we're already confidently tracking in that zone
                if not (predicted_center and predicted_center[0] < 300 and self.tracking_confidence > 0.7):
                    continue
            
            # Skip edges
            if cx < 50 or cx > self.width - 50 or cy < 50 or cy > self.height - 50:
                continue
            
            # Measure event density (key discriminator for drone vs noise)
            distances = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
            tight_count = np.sum(distances < 60)   # Very tight cluster (propellers)
            medium_count = np.sum(distances < 100) # Medium radius
            wide_count = np.sum(distances < 150)   # Wide area
            
            if tight_count < 30 or wide_count < 60:
                continue
            
            # Multi-level density: drone has VERY tight core
            core_density = tight_count / (medium_count + 1)
            overall_density = medium_count / (wide_count + 1)
            
            # Drone: high core density (0.4+) AND good overall density (0.5+)
            # Noise: scattered, low core density (<0.3)
            if core_density < 0.25 or overall_density < 0.45:
                continue
            
                        # Multi-level density: drone has VERY tight core
            core_density = tight_count / (medium_count + 1)
            overall_density = medium_count / (wide_count + 1)
            
            # Drone: high core density (0.4+) AND good overall density (0.5+)
            # Noise: scattered, low core density (<0.3)
            if core_density < 0.25 or overall_density < 0.45:
                continue
            
            # SHAPE ANALYSIS: Drone has structured rectangular/elliptical shape
            # Noise is random scattered points with no structure
            nearby_x = x_coords[distances < 100]
            nearby_y = y_coords[distances < 100]
            
            if len(nearby_x) < 40:
                continue
            
            # CONNECTIVITY ANALYSIS: Drone propellers = continuous black blobs
            # Noise = isolated scattered pixels with no neighbors
            # Count pixels that have nearby neighbors (connected regions)
            # Use spatial hashing for speed
            sample_size = min(len(nearby_x), 150)
            sample_indices = np.random.choice(len(nearby_x), sample_size, replace=False) if len(nearby_x) > sample_size else np.arange(len(nearby_x))
            
            connected_count = 0
            for idx in sample_indices:
                px, py = nearby_x[idx], nearby_y[idx]
                # Check if this pixel has neighbors within 5px (part of a blob)
                neighbor_distances = np.sqrt((nearby_x - px)**2 + (nearby_y - py)**2)
                neighbors = np.sum((neighbor_distances > 0) & (neighbor_distances < 5))
                if neighbors >= 2:  # Has at least 2 neighbors = part of continuous region
                    connected_count += 1
            
            connectivity_ratio = connected_count / sample_size
            
            # Drone: high connectivity (0.4+), continuous blobs
            # Noise: low connectivity (<0.3), isolated pixels
            if connectivity_ratio < 0.35:
                continue
            
            # Calculate aspect ratio and compactness
            x_spread = np.std(nearby_x)
            y_spread = np.std(nearby_y)
            
            # SIZE FILTER: Reject small distant objects (airplanes, birds)
            # Drone is close: x_spread 25-80px, y_spread 8-40px typically
            # Distant airplane/bird: much smaller, <15px spread
            min_spread = min(x_spread, y_spread)
            max_spread = max(x_spread, y_spread)
            
            # Reject if too small (distant object like airplane at std=8x3)
            # or too large (noise blob at std=341x281)
            if min_spread < 8 or max_spread < 25 or max_spread > 120:
                continue
            
            # Drone typically has aspect ratio 1.5-4.0 (elongated)
            # Noise is more circular/irregular (ratio close to 1 or very high)
            aspect_ratio = max_spread / (min_spread + 1)
            
            # Skip if shape is too circular (noise) or too elongated (vertical noise streak)
            if aspect_ratio < 1.3 or aspect_ratio > 5.0:
                continue
            
            # Compactness: events concentrated in ellipse vs scattered
            # For structured shape, most events should fit in 2-std ellipse
            x_centered = nearby_x - cx
            y_centered = nearby_y - cy
            normalized_dist = (x_centered / (x_spread + 1))**2 + (y_centered / (y_spread + 1))**2
            within_2std = np.sum(normalized_dist < 4)  # Within 2 std deviations
            compactness = within_2std / len(nearby_x)
            
            # Drone: high compactness (0.7+), noise: scattered (< 0.6)
            if compactness < 0.65:
                continue
            
            # Score: connectivity (most important) + density + compactness + size
            # Larger objects score higher (drone vs distant airplane)
            size_score = (min_spread * max_spread) / 100  # Normalize by typical drone size
            score = (connectivity_ratio * 300) * (core_density * 200) * (overall_density * 100) * activity * compactness * (1 + size_score)
            
            # SIZE CONSISTENCY BONUS: Drone size should be stable across frames
            # Reject if size changes dramatically (switching between drone and airplane)
            if hasattr(self, 'last_size') and self.last_size and self.tracking_confidence > 0.5:
                last_min, last_max = self.last_size
                size_change = abs(min_spread - last_min) + abs(max_spread - last_max)
                if size_change < 20:  # Similar size = likely same object
                    score *= 2.0  # Strong bonus for size consistency
                elif size_change > 60:  # Drastically different size = likely wrong object
                    score *= 0.3  # Penalty for inconsistent size
            
            # TRAJECTORY VALIDATION: Reject candidates that would create unrealistic jumps
            # Drone moves smoothly, max ~50-80px per 50ms frame
            if self.last_drone_center and self.tracking_confidence > 0.4:
                jump_dist = np.sqrt((cx - self.last_drone_center[0])**2 + (cy - self.last_drone_center[1])**2)
                if jump_dist > 200:  # Reject unrealistic jumps
                    continue
            
            # Strong bonus if near prediction (follow the drone!)
            if predicted_center:
                dist = np.sqrt((cx - predicted_center[0])**2 + (cy - predicted_center[1])**2)
                if dist < 200:
                    # VERY strong tracking: once locked, follow the black blobs
                    proximity_multiplier = 1 + 10.0 / (dist / 30 + 1)  # Up to 11x for close matches
                    score *= proximity_multiplier
            
            if score > best_score:
                best_score = score
                best_center = (cx, cy)
                best_size = (min_spread, max_spread)  # Store size for consistency checking
        
        # Update size tracking if we found a good detection
        if best_center and 'best_size' in locals():
            self.last_size = best_size
        
        # Fallback: use prediction/velocity, DO NOT use raw event mean (includes distant objects)
        if not best_center:
            # If we have good tracking history, trust the velocity prediction
            if predicted_center and self.tracking_confidence > 0.3:
                best_center = predicted_center
                self.tracking_confidence *= 0.7  # Decay confidence when using prediction
            # If we have velocity but lost tracking, extrapolate
            elif self.drone_velocity and self.last_drone_center:
                vx, vy = self.drone_velocity
                best_center = (self.last_drone_center[0] + vx, self.last_drone_center[1] + vy)
                self.tracking_confidence = 0.2
            # No tracking history - use safe default (center-right region)
            else:
                best_center = (self.width * 0.6, self.height * 0.5)
                self.tracking_confidence = 0
        
        # POST-VALIDATION: Reject noise zone detections unless confidently tracking there
        # This catches any detections that slipped through the filters
        if best_center:
            cx_final, cy_final = best_center
            if cx_final < 200:
                # Check if we should trust this left-zone detection
                allow_left = False
                if self.last_drone_center and self.last_drone_center[0] < 250 and self.tracking_confidence > 0.6:
                    allow_left = True
                
                if not allow_left:
                    # Reject - use velocity prediction or clean region
                    if self.drone_velocity and self.last_drone_center:
                        vx, vy = self.drone_velocity
                        best_center = (self.last_drone_center[0] + vx, self.last_drone_center[1] + vy)
                        self.tracking_confidence = 0.2
                    else:
                        # Use clean region
                        clean_mask = x_coords > 250
                        if np.sum(clean_mask) > 100:
                            best_center = (np.mean(x_coords[clean_mask]), np.mean(y_coords[clean_mask]))
                            self.tracking_confidence = 0
        
        if best_center:
            # Update tracking confidence based on movement
            if self.last_drone_center:
                movement = np.sqrt((best_center[0] - self.last_drone_center[0])**2 + 
                                 (best_center[1] - self.last_drone_center[1])**2)
                if movement < 150:
                    self.tracking_confidence = min(1.0, self.tracking_confidence + 0.2)
                elif movement > 400:  # Large jump = likely lost tracking
                    self.tracking_confidence = 0.3
                else:
                    self.tracking_confidence = max(0.3, self.tracking_confidence - 0.1)
            else:
                self.tracking_confidence = 0.5
        
        # Update velocity smoothly
        if self.last_drone_center:
            vx = best_center[0] - self.last_drone_center[0]
            vy = best_center[1] - self.last_drone_center[1]
            if self.drone_velocity:
                prev_vx, prev_vy = self.drone_velocity
                vx = 0.7 * vx + 0.3 * prev_vx
                vy = 0.7 * vy + 0.3 * prev_vy
            self.drone_velocity = (vx, vy)
        
        self.last_drone_center = best_center
        center_x, center_y = best_center
        
        # Find propellers
        propellers = self._find_propeller_clusters(x_coords, y_coords, best_center)
        
        # Create bounding box
        if len(propellers) >= 2:
            prop_x = [p[0] for p in propellers]
            prop_y = [p[1] for p in propellers]
            min_x, max_x = min(prop_x) - 40, max(prop_x) + 40
            min_y, max_y = min(prop_y) - 40, max(prop_y) + 40
            
            bbox_mask = ((x_coords >= min_x) & (x_coords <= max_x) &
                        (y_coords >= min_y) & (y_coords <= max_y))
            filtered_x = x_coords[bbox_mask]
            filtered_y = y_coords[bbox_mask]
            
            drone_info = {
                'center': best_center,
                'propellers': propellers,
                'bbox': (min_x, min_y, max_x, max_y),
                'num_propellers': len(propellers),
                'tracking_confidence': self.tracking_confidence
            }
        else:
            # Circular fallback
            distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            roi_mask = distances < 150
            filtered_x = x_coords[roi_mask]
            filtered_y = y_coords[roi_mask]
            
            drone_info = {
                'center': best_center,
                'propellers': [],
                'bbox': (center_x - 150, center_y - 150, center_x + 150, center_y + 150),
                'num_propellers': 0,
                'tracking_confidence': self.tracking_confidence
            }
        
        if len(filtered_x) < 30:
            return x_coords, y_coords, {}
            
        return filtered_x, filtered_y, drone_info
    
    def should_capture(self, current_time_us: int) -> bool:
        """Check if we should capture this frame based on delta."""
        return current_time_us >= self.next_capture_time
    
    def capture_frame(
        self,
        frame: np.ndarray,
        timestamp_us: int,
        label: str = "unlabeled",
        angle: float = 0.0,
        x_coords: np.ndarray = None,
        y_coords: np.ndarray = None,
        num_events: int = 0,
        drone_info: dict = None
    ) -> None:
        """Save frame and metadata."""
        # Generate filename
        filename = f"frame_{self.frame_counter:06d}.png"
        filepath = self.frames_dir / filename
        
        # Save frame
        cv2.imwrite(str(filepath), frame)
        
        # Calculate event statistics with proper drone detection
        event_stats = {}
        if x_coords is not None and len(x_coords) > 0:
            # Filter events to focus on drone region (remove background noise)
            drone_x, drone_y, detected_info = self._detect_drone_region(x_coords, y_coords)
            
            # Check if drone is out of frame
            if detected_info.get('out_of_frame', False):
                event_stats = {
                    "num_events": int(num_events),
                    "num_filtered_events": 0,
                    "out_of_frame": True,
                    "tracking_confidence": 0.0
                }
            elif len(drone_x) > 0:
                event_stats = {
                    "num_events": int(num_events),
                    "num_filtered_events": int(len(drone_x)),
                    "x_mean": float(np.mean(drone_x)),
                    "y_mean": float(np.mean(drone_y)),
                    "x_std": float(np.std(drone_x)),
                    "y_std": float(np.std(drone_y)),
                    "x_min": int(np.min(drone_x)),
                    "x_max": int(np.max(drone_x)),
                    "y_min": int(np.min(drone_y)),
                    "y_max": int(np.max(drone_y))
                }
                
                # Add drone geometry info if available
                if detected_info:
                    if 'propellers' in detected_info and detected_info['propellers']:
                        event_stats['propellers'] = [
                            {'x': float(p[0]), 'y': float(p[1])} 
                            for p in detected_info['propellers']
                        ]
                    if 'bbox' in detected_info:
                        bbox = detected_info['bbox']
                        event_stats['bbox'] = {
                            'x_min': float(bbox[0]),
                            'y_min': float(bbox[1]),
                            'x_max': float(bbox[2]),
                            'y_max': float(bbox[3])
                        }
                    if 'tracking_confidence' in detected_info:
                        event_stats['tracking_confidence'] = float(detected_info['tracking_confidence'])
                    if 'num_propellers' in detected_info:
                        event_stats['num_propellers'] = int(detected_info['num_propellers'])
                    if 'fallback' in detected_info:
                        event_stats['fallback_detection'] = True
            else:
                # Fallback if no dense region found
                event_stats = {
                    "num_events": int(num_events),
                    "num_filtered_events": 0,
                    "x_mean": float(np.mean(x_coords)),
                    "y_mean": float(np.mean(y_coords)),
                    "x_std": float(np.std(x_coords)),
                    "y_std": float(np.std(y_coords)),
                    "x_min": int(np.min(x_coords)),
                    "x_max": int(np.max(x_coords)),
                    "y_min": int(np.min(y_coords)),
                    "y_max": int(np.max(y_coords))
                }
        
        # Store metadata
        frame_metadata = {
            "frame_id": self.frame_counter,
            "filename": filename,
            "timestamp_us": int(timestamp_us),
            "timestamp_s": timestamp_us / 1e6,
            "label": label,
            "angle_deg": float(angle),
            "event_stats": event_stats
        }
        
        self.metadata["frames"].append(frame_metadata)
        
        # Update counters
        self.next_capture_time = timestamp_us + self.delta_us
        self.frame_counter += 1
        
        print(f"Captured frame {self.frame_counter}: {filename} (t={timestamp_us/1e6:.3f}s, label={label})")
    
    def save_metadata(self) -> None:
        """Save metadata JSON file."""
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"\nSaved metadata to {metadata_path}")
        print(f"Total frames captured: {self.frame_counter}")
    
    def calculate_reference_features(self) -> dict:
        """Calculate reference features from captured frames (for idle drone)."""
        if not self.metadata["frames"]:
            return {}
        
        # Aggregate event statistics (skip out-of-frame frames)
        all_x_means = [f["event_stats"]["x_mean"] for f in self.metadata["frames"] 
                       if f["event_stats"] and "x_mean" in f["event_stats"]]
        all_y_means = [f["event_stats"]["y_mean"] for f in self.metadata["frames"] 
                       if f["event_stats"] and "y_mean" in f["event_stats"]]
        all_x_stds = [f["event_stats"]["x_std"] for f in self.metadata["frames"] 
                      if f["event_stats"] and "x_std" in f["event_stats"]]
        all_y_stds = [f["event_stats"]["y_std"] for f in self.metadata["frames"] 
                      if f["event_stats"] and "y_std" in f["event_stats"]]
        
        reference = {
            "centroid": {
                "x": float(np.mean(all_x_means)) if all_x_means else 0.0,
                "y": float(np.mean(all_y_means)) if all_y_means else 0.0
            },
            "spread": {
                "x": float(np.mean(all_x_stds)) if all_x_stds else 0.0,
                "y": float(np.mean(all_y_stds)) if all_y_stds else 0.0
            }
        }
        
        return reference


def get_window(
    event_words: np.ndarray,
    time_order: np.ndarray,
    win_start: int,
    win_stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract event window."""
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
    """Generate frame from event window."""
    x_coords, y_coords, polarities_on = window
    frame = np.full((height, width, 3), base_color, np.uint8)
    frame[y_coords[polarities_on], x_coords[polarities_on]] = on_color
    frame[y_coords[~polarities_on], x_coords[~polarities_on]] = off_color

    return frame


def estimate_angle_deviation(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    reference: dict,
    drone_info: dict = None
) -> float:
    """
    Estimate angle deviation from reference (idle) position.
    Uses propeller geometry and spread changes to estimate viewing angle.
    """
    if len(x_coords) < 10 or not reference:
        return 0.0
    
    # Current measurements
    current_x = np.mean(x_coords)
    current_y = np.mean(y_coords)
    current_x_std = np.std(x_coords)
    current_y_std = np.std(y_coords)
    
    # Reference values
    ref_x = reference.get("centroid", {}).get("x", current_x)
    ref_y = reference.get("centroid", {}).get("y", current_y)
    ref_x_std = reference.get("spread", {}).get("x", current_x_std)
    ref_y_std = reference.get("spread", {}).get("y", current_y_std)
    
    # Method 1: Aspect ratio change (most reliable for angle)
    # When face-on (0°), drone appears roughly circular
    # When tilted, it becomes elongated
    if ref_x_std > 0 and ref_y_std > 0:
        current_aspect = min(current_x_std, current_y_std) / max(current_x_std, current_y_std)
        ref_aspect = min(ref_x_std, ref_y_std) / max(ref_x_std, ref_y_std)
        
        # Convert aspect ratio to angle
        # aspect = 1.0 → 0°, aspect = 0.0 → 90°
        angle_from_aspect = (1.0 - current_aspect) * 90.0
    else:
        angle_from_aspect = 0.0
    
    # Method 2: Propeller geometry (if available)
    angle_from_propellers = None
    if drone_info and 'propellers' in drone_info and len(drone_info['propellers']) >= 3:
        propellers = drone_info['propellers']
        
        # Calculate spread of propeller positions
        # Propellers are stored as tuples (x, y)
        prop_x = [p[0] for p in propellers]
        prop_y = [p[1] for p in propellers]
        
        prop_x_range = max(prop_x) - min(prop_x)
        prop_y_range = max(prop_y) - min(prop_y)
        
        if prop_x_range > 0 and prop_y_range > 0:
            # Aspect ratio of propeller layout
            prop_aspect = min(prop_x_range, prop_y_range) / max(prop_x_range, prop_y_range)
            angle_from_propellers = (1.0 - prop_aspect) * 75.0  # Propellers give angle up to ~75°
    
    # Method 3: Centroid displacement (less reliable)
    shift_x = abs(current_x - ref_x)
    shift_y = abs(current_y - ref_y)
    total_shift = np.sqrt(shift_x**2 + shift_y**2)
    angle_from_shift = min(30.0, (total_shift / 80.0) * 30.0)
    
    # Combine methods with smart weighting
    if angle_from_propellers is not None:
        # Propeller geometry is most accurate
        estimated_angle = 0.6 * angle_from_propellers + 0.3 * angle_from_aspect + 0.1 * angle_from_shift
    else:
        # Fallback to aspect ratio and shift
        estimated_angle = 0.7 * angle_from_aspect + 0.3 * angle_from_shift
    
    # Clamp to reasonable range
    estimated_angle = max(0.0, min(90.0, estimated_angle))
    
    return estimated_angle


def process_recording(
    dat_path: str,
    output_dir: str,
    delta_ms: float,
    label: str,
    reference: dict = None,
    window_ms: float = 10.0,
    speed: float = 1.0
) -> dict:
    """
    Process a .dat recording and generate dataset.
    
    Args:
        dat_path: Path to .dat file
        output_dir: Output directory for dataset
        delta_ms: Time interval between captures in milliseconds
        label: Label for this recording (e.g., "idle", "moving")
        reference: Reference features from idle drone (for angle estimation)
        window_ms: Event window duration in milliseconds
        speed: Playback speed
    
    Returns:
        Dictionary with reference features (for idle recordings)
    """
    print(f"\n{'='*60}")
    print(f"Processing: {dat_path}")
    print(f"Label: {label}")
    print(f"Output: {output_dir}")
    print(f"Capture interval: {delta_ms}ms")
    print(f"{'='*60}\n")
    
    # Create dataset generator
    dataset = DatasetGenerator(output_dir, delta_ms=delta_ms)
    
    # Load source
    src = DatFileSource(
        dat_path, 
        width=1280, 
        height=720, 
        window_length_us=window_ms * 1000
    )
    
    pacer = Pacer(speed=speed, force_speed=False)
    
    # Process frames
    for batch_range in pacer.pace(src.ranges()):
        if dataset.should_capture(batch_range.end_ts_us):
            # Extract window
            window = get_window(
                src.event_words,
                src.order,
                batch_range.start,
                batch_range.stop,
            )
            
            x_coords, y_coords, polarities = window
            num_events = batch_range.stop - batch_range.start
            
            # Filter events to detect drone region with geometry info
            filtered_x, filtered_y, drone_info = dataset._detect_drone_region(x_coords, y_coords)
            
            # Generate frame (uses all events for visualization)
            frame = get_frame(window)
            
            # Estimate angle using filtered drone events and geometry
            angle = 0.0
            if reference:
                angle = estimate_angle_deviation(filtered_x, filtered_y, reference, drone_info)
            
            # Capture frame (will re-filter internally for metadata)
            dataset.capture_frame(
                frame,
                batch_range.end_ts_us,
                label=label,
                angle=angle,
                x_coords=x_coords,
                y_coords=y_coords,
                num_events=num_events,
                drone_info=drone_info
            )
            
            # Debug output for first few frames
            if dataset.frame_counter <= 3:
                if drone_info and drone_info.get('out_of_frame', False):
                    print(f"  -> Drone OUT OF FRAME (no detection)")
                else:
                    print(f"  -> Filtered {len(x_coords)} events to {len(filtered_x)} drone events")
                    if len(filtered_x) > 0:
                        print(f"  -> Drone centroid: ({np.mean(filtered_x):.1f}, {np.mean(filtered_y):.1f})")
                    if drone_info and 'num_propellers' in drone_info:
                        print(f"  -> Detected {drone_info['num_propellers']} propellers")
                    if drone_info and 'tracking_confidence' in drone_info:
                        print(f"  -> Tracking confidence: {drone_info['tracking_confidence']:.2f}")
                    if drone_info and drone_info.get('fallback_detection', False):
                        print(f"  -> Using fallback detection (velocity prediction)")
                if reference:
                    ref_cx = reference.get("centroid", {}).get("x", 0)
                    ref_cy = reference.get("centroid", {}).get("y", 0)
                    print(f"  -> Reference centroid: ({ref_cx:.1f}, {ref_cy:.1f})")
                    if not (drone_info and drone_info.get('out_of_frame', False)):
                        print(f"  -> Estimated angle: {angle:.2f}°")
    
    # Calculate reference features (for idle drone)
    reference_features = dataset.calculate_reference_features()
    
    # Add reference to metadata
    dataset.metadata["reference_features"] = reference_features
    dataset.metadata["recording_info"] = {
        "source_file": str(Path(dat_path).name),
        "label": label
    }
    
    # Save metadata
    dataset.save_metadata()
    
    return reference_features


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate dataset from event camera recordings"
    )
    parser.add_argument(
        "dat_file",
        help="Path to .dat file"
    )
    parser.add_argument(
        "--output",
        default="./dataset",
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=100,
        help="Time interval between frame captures in milliseconds (default: 100ms)"
    )
    parser.add_argument(
        "--label",
        default="unlabeled",
        help="Label for this recording (e.g., 'idle', 'moving')"
    )
    parser.add_argument(
        "--reference",
        help="Path to reference metadata JSON (from idle recording)"
    )
    parser.add_argument(
        "--window",
        type=float,
        default=10,
        help="Event window duration in milliseconds"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed"
    )
    
    args = parser.parse_args()
    
    # Load reference if provided
    reference = None
    if args.reference:
        with open(args.reference, 'r') as f:
            ref_metadata = json.load(f)
            reference = ref_metadata.get("reference_features", {})
            print(f"Loaded reference from: {args.reference}")
    
    # Process recording
    reference_features = process_recording(
        args.dat_file,
        args.output,
        args.delta,
        args.label,
        reference=reference,
        window_ms=args.window,
        speed=args.speed
    )
    
    print(f"\n{'='*60}")
    print("Dataset generation complete!")
    print(f"{'='*60}")
    
    if reference_features:
        print("\nReference features (use for moving drone):")
        print(json.dumps(reference_features, indent=2))


if __name__ == "__main__":
    main()