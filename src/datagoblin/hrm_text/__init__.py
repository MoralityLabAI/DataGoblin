"""Optional HRM-Text tracing and calibration interfaces."""

from .availability import HRMAvailability, detect_hrm_availability
from .instrumented_forward import TraceOptions, run_instrumented_forward
from .trace_protocol import TRACE_ORDER, TraceableRecurrentModel

__all__ = [
    "TRACE_ORDER",
    "HRMAvailability",
    "TraceOptions",
    "TraceableRecurrentModel",
    "detect_hrm_availability",
    "run_instrumented_forward",
]
