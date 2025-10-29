from scripts.utils.utils_aux import iou


class OperationFSM:
    """
    Finite State Machine to track operations:
    PickUp -> ProbePass -> Marking -> Place
    """

    def __init__(self, iou_thresh=0.3):
        self.state = "Idle"
        self.start_time = None
        self.probe_passes = 0
        self.markings = 0
        self.metrics = {
            "total_operations": 0,
            "durations": [],
            "with_probe": 0,
            "with_marking": 0
        }
        self.iou_thresh = iou_thresh
        self.last_event = None
        self.last_event_frame = -999

    def emit_event(self, event, frame_idx, cooldown=10):
        """
        Emit an event, but suppress duplicates if they occur
        within 'cooldown' frames of the last identical event.
        """
        if self.last_event == event and (frame_idx - self.last_event_frame) < cooldown:
            return None  # suppress duplicate
        self.last_event = event
        self.last_event_frame = frame_idx
        return event


    def update(self, hands, pieces, probes, markers, frame_idx, fps):
        """
        Update FSM state given current detections.
        hands, pieces, probes, markers are lists of [x1,y1,x2,y2].
        """
        if self.state == "Idle":
            # Detect PickUp: hand overlaps with piece
            for h in hands:
                for p in pieces:
                    if iou(h, p) > self.iou_thresh:
                        self.state = "PickUp"
                        self.start_time = frame_idx / fps
                        self.probe_passes = 0
                        self.markings = 0
                        return self.emit_event("PickUp", frame_idx)

        elif self.state == "PickUp":
            # Detect probe entering piece
            for pr in probes:
                for p in pieces:
                    if iou(pr, p) > self.iou_thresh:
                        self.state = "ProbePass"
                        return self.emit_event("ProbePass", frame_idx)

        elif self.state == "ProbePass":
            # Count probe passes
            for pr in probes:
                for p in pieces:
                    if iou(pr, p) > self.iou_thresh:
                        probepass_event = self.emit_event("ProbePass", frame_idx)
                        if probepass_event:  # only increment if not suppressed
                            self.probe_passes += 1
                            return probepass_event
                # Only advance after having a 2 or more number of probe_passes
                if self.probe_passes >= 2:
                    # Detect markings
                    for m in markers:
                        for p in pieces:
                            if iou(m, p) > self.iou_thresh:
                                self.state = "Marking"
                                marking_event = self.emit_event("Marking", frame_idx)
                                if marking_event:
                                    self.markings += 1
                                return marking_event

        elif self.state == "Marking":
            # Detect markings
            for m in markers:
                for p in pieces:
                    if iou(m, p) > self.iou_thresh:
                        marking_event = self.emit_event("Marking", frame_idx)
                        if marking_event:
                            self.markings += 1
                        return marking_event
            # Detect placing: piece disappears or hand leaves screen
            if len(hands) == 1:
                self.state = "Place"
                return self.emit_event("Place", frame_idx)

        elif self.state == "Place":
            # Operation finished
            end_time = frame_idx / fps
            duration = end_time - self.start_time if self.start_time else 0
            self.metrics["total_operations"] += 1
            self.metrics["durations"].append(duration)
            if self.probe_passes > 0:
                self.metrics["with_probe"] += 1
            if self.markings > 0:
                self.metrics["with_marking"] += 1
            self.state = "Idle"
            return "Completed"

        return None

    def summary(self):
        """
        Return computed metrics.
        """
        total = self.metrics["total_operations"]
        avg_duration = sum(self.metrics["durations"]) / total if total > 0 else 0
        pct_probe = (self.metrics["with_probe"] / total * 100) if total > 0 else 0
        pct_marking = (self.metrics["with_marking"] / total * 100) if total > 0 else 0

        return {
            "total_operations": total,
            "average_duration": avg_duration,
            "percentage_with_probe": pct_probe,
            "percentage_with_marking": pct_marking
        }


if __name__ == "__main__":
    # Synthetic test of the FSM with more frames and bounding boxes
    from time import sleep

    fsm = OperationFSM(iou_thresh=0.01)
    fps = 30
    frame_idx = 0

    # Define dummy bounding boxes [x1, y1, x2, y2]
    left_hand = [10, 10, 50, 50]
    right_hand = [70, 10, 110, 50]
    piece = [20, 20, 60, 60]
    probe = [75, 20, 85, 40]   # starts near right hand
    marker = [75, 20, 85, 40]   # same region, different tool

    # --- 1. PickUp: left hand overlaps piece ---
    frame_idx += 1
    event = fsm.update([left_hand, right_hand], [piece], [], [], frame_idx, fps)
    print(f"Frame {frame_idx}: {event}, State={fsm.state}")

    # --- 2. ProbePass: probe moves into piece (simulate overlap) ---
    probe_on_piece = [25, 25, 35, 35]  # inside piece
    frame_idx += 1
    event = fsm.update([left_hand, right_hand], [piece], [probe_on_piece], [], frame_idx, fps)
    print(f"Frame {frame_idx}: {event}, State={fsm.state}")

    # Keep probe overlapping for another frame (count multiple passes)
    frame_idx += 1
    event = fsm.update([left_hand, right_hand], [piece], [probe_on_piece], [], frame_idx, fps)
    print(f"Frame {frame_idx}: {event}, State={fsm.state}")

    # --- 3. Marking: marker overlaps piece ---
    marker_on_piece = [30, 30, 40, 40]  # inside piece
    frame_idx += 1
    event = fsm.update([left_hand, right_hand], [piece], [], [marker_on_piece], frame_idx, fps)
    print(f"Frame {frame_idx}: {event}, State={fsm.state}")

    # --- 4. Place: piece disappears, only left hand remains ---
    frame_idx += 1
    event = fsm.update([left_hand], [], [], [], frame_idx, fps)
    print(f"Frame {frame_idx}: {event}, State={fsm.state}")

    # --- 5. Completed: no hands, no piece ---
    frame_idx += 1
    event = fsm.update([], [], [], [], frame_idx, fps)
    print(f"Frame {frame_idx}: {event}, State={fsm.state}")

    # Print summary metrics
    print("Final metrics:", fsm.summary())
