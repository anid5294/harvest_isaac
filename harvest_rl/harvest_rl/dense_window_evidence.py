"""Durable raw native windows before any task reader/abort; no simulator imports."""
from dataclasses import asdict,is_dataclass
import json,os
import math
from pathlib import Path

class DurableNativeWindows:
    def __init__(self,path):
        self.path=Path(path)
        self.stream=self.path.open('x')
        self.records=0
        self.last_sample=None

    def __call__(self,window):
        if self.stream.closed:raise ValueError('Native-window evidence writer closed')
        if not is_dataclass(window) or type(window.sample) is not int:
            raise ValueError('Require an actual typed native contact window')
        if self.last_sample is not None and window.sample!=self.last_sample+1:
            raise ValueError('Duplicate/nonconsecutive raw window; no silent overwrite')
        raw=json.dumps(asdict(window),allow_nan=False,separators=(',',':'))+'\n'
        self.stream.write(raw)
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.last_sample=window.sample
        self.records+=1

    def close(self):
        if not self.stream.closed:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()

def startup_capture_summary(window, *, records, control_steps, actions, detached_ids):
    """Capture validity and physical stability remain separate outcomes."""
    if (window is None or not window.complete or window.physics_steps!=4
            or window.sample!=1 or records!=1 or control_steps!=1
            or len(actions)!=14 or any(not math.isfinite(x) or x!=0 for x in actions)):
        raise ValueError('Startup capture requires exactly one complete zero-action1control/4physics window')
    return dict(startup_capture_complete=True,initial_inventory_stable=not detached_ids,
                detached_ids=list(detached_ids),neutral_control_steps=1,physics_steps=4,
                learned_policy_actions=0,raw_window_records=1)
