"""
AXI4-Lite register map for the AV SoC bridge.

Defines the address space, reset values, and access permissions for every
register that the software mitigation engine uses to control hardware
detectors.  The bridge module reads/writes through this map; the FPGA
driver reads from it to get dynamically configured thresholds.

Address map (32-bit words):
  0x00  GPS_ENABLE          RW  1=GPS active, 0=GPS isolated
  0x04  V2X_GAIN            RW  V2X gain setting (0=max, 255=min)
  0x08  SAFE_STOP_CMD       RW  1=initiate safe stop, 0=clear
  0x0C  CAN_MIN_PERIOD_0    RW  min period for CAN ID 0x0C0 (cycles)
  0x10  CAN_MIN_PERIOD_1    RW  min period for CAN ID 0x0D0 (cycles)
  0x14  CAN_MIN_PERIOD_2    RW  min period for CAN ID 0x110 (cycles)
  0x18  CAN_MIN_PERIOD_3    RW  min period for CAN ID 0x320 (cycles)
  0x1C  CAN_RATE_FACTOR     RW  bus rate multiplier (fixed point, 8.8)
  0x20  PERCEPTION_MASK     RW  1=mask low-confidence regions, 0=normal
  0x24  MITIGATION_STATE    R   current mitigation state (read-only)
  0x28  THREAT_SCORE        R   current T_joint (read-only, 8.8 fixed)
  0x2C  HW_ALERT_STATUS     R   which HW detectors have fired (bitmap)
  0x30  VERSION             R   design version (read-only)
"""

# Register definitions: (address, name, reset_value, access, description)
REGISTERS = {
    0x00: ("GPS_ENABLE",          1,   "RW", "GPS receiver enable (1=on, 0=isolated)"),
    0x04: ("V2X_GAIN",            0,   "RW", "V2X radio gain (0=max, 255=silenced)"),
    0x08: ("SAFE_STOP_CMD",       0,   "RW", "Safe-stop command (1=pull over, 0=clear)"),
    0x0C: ("CAN_MIN_PERIOD_0",    80,  "RW", "Min period CAN ID 0x0C0 (cycles)"),
    0x10: ("CAN_MIN_PERIOD_1",    80,  "RW", "Min period CAN ID 0x0D0 (cycles)"),
    0x14: ("CAN_MIN_PERIOD_2",    160, "RW", "Min period CAN ID 0x110 (cycles)"),
    0x18: ("CAN_MIN_PERIOD_3",    800, "RW", "Min period CAN ID 0x320 (cycles)"),
    0x1C: ("CAN_RATE_FACTOR",     320, "RW", "Bus rate factor (8.8 fixed, 1.25x=320)"),
    0x20: ("PERCEPTION_MASK",     0,   "RW", "Perception mask (1=filter, 0=normal)"),
    0x24: ("MITIGATION_STATE",    0,   "R",  "Current mitigation state (enum)"),
    0x28: ("THREAT_SCORE",        0,   "R",  "Current T_joint (8.8 fixed point)"),
    0x2C: ("HW_ALERT_STATUS",     0,   "R",  "HW alert bitmap (bit0=timing, bit1=unknown)"),
    0x30: ("VERSION",             0x0102, "R", "Design version (1.2)"),
}

# Mitigation state enum values (for MITIGATION_STATE register)
STATE_ENUM = {
    "NORMAL":          0,
    "MONITORING":      1,
    "GPS_ISOLATED":    2,
    "V2X_DEGRADED":    3,
    "PERCEPTION_MASKED": 4,
    "CAN_SAFE_STOP":   5,
    "FULL_FALLBACK":   6,
}

# CAN ID to register address mapping
CAN_ID_MAP = {
    0x0C0: 0x0C,
    0x0D0: 0x10,
    0x110: 0x14,
    0x320: 0x18,
}
