from src.dip.shape_analysis import compute_shape_similarity
from config.settings import (
    CONF_MIN,
    CONF_HIGH,
    WEIGHT_AI,
    WEIGHT_SHAPE,
    WEIGHTED_SCORE_MIN,
    FRAME_ACCUMULATION_MIN,
    DASHBOARD_ACTIVE
)

class ThreatVerifier:
    """
    Implements the False-Positive Rejection Architecture flowchart:
    
    1. Confidence Score evaluation:
       - < 60%           -> Skip Frame (Ignored as noise)
       - > 90%           -> High confidence (Bypasses shape similarity)
       - 60% - 90%       -> ROI extracted -> Shape Similarity -> Weighted Scoring
    2. Weighted Scoring System:
       - Score = (0.60 * AI_Conf + 0.40 * Shape_Score)
       - < 75%           -> Skip Frame
       - >= 75% (> 74%)  -> Advances to Frame Accumulation
    3. Frame Accumulation:
       - < 4 Frames      -> Skip Frame (Holds & accumulates temporal consistency)
       - > 3 Frames      -> Advances to Dashboard Check
    4. Dashboard Check:
       - If Armed        -> Detection Confirm -> Triggers Alert (MQTT / Buzzer)
    """
    def __init__(self, alert_manager=None):
        self.alert_manager = alert_manager
        self.consecutive_frames = 0
        self.dashboard_active = True
        self.last_confirmed_threat = None
        self.conf_min = CONF_MIN
        self.conf_high = CONF_HIGH
        self.weight_ai = WEIGHT_AI
        self.weight_shape = WEIGHT_SHAPE
        self.weighted_score_min = WEIGHTED_SCORE_MIN
        self.frame_accum_min = FRAME_ACCUMULATION_MIN

    def verify_candidates(self, candidates, frame):
        """
        Processes candidate detections through the flowchart verification pipeline.
        
        Returns:
          verified_threats: list of dicts with full verification metadata:
            {
              'box': (x1, y1, x2, y2),
              'label': str,
              'ai_conf': float,
              'shape_score': float,
              'weighted_score': float,
              'status': 'SKIP' | 'ACCUMULATING' | 'CONFIRMED'
            }
          is_confirmed (bool)
        """
        verified_threats = []
        best_candidate = None
        best_score = 0.0

        for cand in candidates:
            ai_conf = cand['conf']
            roi = cand['roi']
            box = cand['box']
            label = cand['label']

            # 1. Confidence Score Check
            if ai_conf < self.conf_min:
                # Below minimum -> Skip Frame
                continue

            if ai_conf >= self.conf_high:
                # Above high confidence -> direct pass to Frame Accumulation
                shape_score = 1.0
                weighted_score = ai_conf
                passes_scoring = True
            else:
                # Intermediate bracket -> ROI Shape Similarity + Weighted Scoring
                shape_score = compute_shape_similarity(roi)
                weighted_score = (self.weight_ai * ai_conf) + (self.weight_shape * shape_score)
                # Weighted Scoring check
                passes_scoring = (weighted_score >= self.weighted_score_min)

            if not passes_scoring:
                # Failed weighted scoring -> Skip Frame
                continue

            if weighted_score > best_score:
                best_score = weighted_score
                best_candidate = {
                    'box': box,
                    'label': label,
                    'ai_conf': ai_conf,
                    'shape_score': shape_score,
                    'weighted_score': weighted_score
                }

        # 2. Frame Accumulation
        if best_candidate is not None:
            self.consecutive_frames += 1
        else:
            # Decay counter if no valid detection in this frame
            self.consecutive_frames = max(0, self.consecutive_frames - 1)

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
                status = f"ACCUMULATING ({self.consecutive_frames}/{FRAME_ACCUMULATION_MIN})"

            best_candidate['status'] = status
            best_candidate['accum_count'] = self.consecutive_frames
            verified_threats.append(best_candidate)

        return verified_threats, is_confirmed
