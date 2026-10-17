from config.settings import CONF_MIN, FRAME_ACCUMULATION_MIN

class ThreatVerifier:
    """
    Verification pipeline with Post-DIP shape analysis removed, preserving Frame Accumulation:
    
    1. Confidence Score evaluation:
       - < conf_min -> Skip Frame (Ignored as noise)
       - >= conf_min -> Advances directly to Frame Accumulation (Post-DIP bypassed)
    2. Frame Accumulation:
       - < frame_accum_min -> Accumulating (Ensures temporal consistency)
       - >= frame_accum_min -> Advances to Dashboard Check
    3. Dashboard Check:
       - If Armed -> Detection Confirm -> Triggers Alert (MQTT / Buzzer)
    """
    def __init__(self, alert_manager=None):
        self.alert_manager = alert_manager
        self.consecutive_frames = 0
        self.miss_streak = 0          # Tracks consecutive missed frames before decaying counter
        self.dashboard_active = True
        self.last_confirmed_threat = None
        self.conf_min = CONF_MIN
        self.frame_accum_min = FRAME_ACCUMULATION_MIN

    def verify_candidates(self, candidates, frame):
        """
        Processes candidate detections:
        Bypasses post-DIP shape similarity, using raw AI confidence directly
        through temporal Frame Accumulation.
        
        Returns:
          verified_threats: list of dicts with verification metadata:
            {
              'box': (x1, y1, x2, y2),
              'label': str,
              'ai_conf': float,
              'weighted_score': float, # equal to ai_conf
              'status': 'ACCUMULATING' | 'CONFIRMED' | 'DASHBOARD_INACTIVE',
              'accum_count': int,
              'roi': np.ndarray
            }
          is_confirmed (bool)
        """
        verified_threats = []
        best_candidate = None
        best_score = 0.0

        for cand in candidates:
            ai_conf = cand['conf']
            roi = cand.get('roi')
            box = cand['box']
            label = cand['label']

            # 1. Confidence Score Check
            if ai_conf < self.conf_min:
                # Below minimum -> Skip Frame
                continue

            # Directly use AI confidence (post-DIP shape similarity removed)
            if ai_conf > best_score:
                best_score = ai_conf
                best_candidate = {
                    'box': box,
                    'label': label,
                    'ai_conf': ai_conf,
                    'weighted_score': ai_conf,
                    'roi': roi
                }

        # 2. Frame Accumulation
        if best_candidate is not None:
            self.consecutive_frames += 1
            self.miss_streak = 0  # Reset miss streak on any valid detection
        else:
            # Allow up to 2 consecutive missed frames before decaying counter.
            # This handles motion blur, brief occlusion, and frame drops.
            self.miss_streak += 1
            if self.miss_streak >= 2:
                self.consecutive_frames = max(0, self.consecutive_frames - 1)
                self.miss_streak = 0

        is_confirmed = False
        if best_candidate is not None:
            # Check frame accumulation threshold
            if self.consecutive_frames >= self.frame_accum_min:
                # 3. Dashboard Check
                if self.dashboard_active:
                    status = "CONFIRMED"
                    is_confirmed = True
                    self.last_confirmed_threat = best_candidate['label']
                    # 4. Trigger MQTT / Buzzer
                    if self.alert_manager is not None:
                        self.alert_manager.on_detection_confirm(
                            frame, best_candidate['label'], best_candidate['weighted_score']
                        )
                else:
                    status = "DASHBOARD_INACTIVE"
            else:
                status = f"ACCUMULATING ({self.consecutive_frames}/{self.frame_accum_min})"

            best_candidate['status'] = status
            best_candidate['accum_count'] = self.consecutive_frames
            verified_threats.append(best_candidate)

        return verified_threats, is_confirmed
