"""
SoC bridge simulator: software-to-hardware register interface.

Simulates an AXI4-Lite bus bridge that the mitigation engine writes to
and the FPGA/CAN drivers read from.  Maintains a register file in memory
and provides read/write operations with proper access control.

Usage:
    from bridge import SoCBridge
    bridge = SoCBridge()
    bridge.write_reg("GPS_ENABLE", 0)
    bridge.apply_mitigation(mitigation_result)
    gps_enabled = bridge.read_reg("GPS_ENABLE")
    can_period = bridge.get_can_min_period(0x0C0)
"""
from typing import Dict, Optional
from axi_registers import REGISTERS, STATE_ENUM, CAN_ID_MAP


class SoCBridge:
    """Simulated AXI4-Lite register bridge."""

    def __init__(self):
        # Initialize all registers to their reset values
        self.regs: Dict[int, int] = {}
        self.reg_names: Dict[str, int] = {}
        for addr, (name, reset, access, desc) in REGISTERS.items():
            self.regs[addr] = reset
            self.reg_names[name] = addr
        self.write_log: list = []

    def read_reg(self, name_or_addr) -> int:
        """Read a register by name or address."""
        if isinstance(name_or_addr, str):
            addr = self.reg_names.get(name_or_addr)
            if addr is None:
                raise KeyError(f"Unknown register: {name_or_addr}")
        else:
            addr = name_or_addr
        entry = REGISTERS.get(addr)
        if entry is None:
            raise KeyError(f"Unknown address: 0x{addr:02X}")
        return self.regs.get(addr, 0)

    def write_reg(self, name_or_addr, value: int) -> bool:
        """Write a register by name or address.  Returns True if successful."""
        if isinstance(name_or_addr, str):
            addr = self.reg_names.get(name_or_addr)
            if addr is None:
                return False
        else:
            addr = name_or_addr
        entry = REGISTERS.get(addr)
        if entry is None:
            return False
        _, _, access, _ = entry
        if access == "R":
            return False  # read-only register
        old_val = self.regs.get(addr, 0)
        self.regs[addr] = value & 0xFFFFFFFF
        self.write_log.append({
            "address": f"0x{addr:02X}",
            "name": entry[0],
            "old": old_val,
            "new": value,
        })
        return True

    def apply_mitigation(self, mitigation: Dict):
        """Apply all register overrides from a mitigation result dict."""
        overrides = mitigation.get("register_overrides", {})
        for name, value in overrides.items():
            self.write_reg(name, value)

        # Update the read-only status registers
        state_name = mitigation.get("state", "NORMAL")
        state_val = STATE_ENUM.get(state_name, 0)
        self.regs[self.reg_names["MITIGATION_STATE"]] = state_val

        trigger = mitigation.get("trigger", {})
        t_joint = trigger.get("t_joint", 0.0)
        self.regs[self.reg_names["THREAT_SCORE"]] = int(t_joint * 256)  # 8.8 fixed

    def get_can_min_period(self, arb_id: int) -> Optional[int]:
        """Get the current min_period for a CAN arbitration ID."""
        addr = CAN_ID_MAP.get(arb_id)
        if addr is not None:
            return self.regs.get(addr)
        return None

    def get_all_can_periods(self) -> Dict[str, int]:
        """Return all CAN min_period register values."""
        result = {}
        for arb_id, addr in CAN_ID_MAP.items():
            name = REGISTERS[addr][0]
            result[f"0x{arb_id:03X}"] = self.regs.get(addr, 0)
        return result

    def get_gps_enabled(self) -> bool:
        return self.regs.get(self.reg_names["GPS_ENABLE"], 1) == 1

    def get_v2x_gain(self) -> int:
        return self.regs.get(self.reg_names["V2X_GAIN"], 0)

    def get_safe_stop(self) -> bool:
        return self.regs.get(self.reg_names["SAFE_STOP_CMD"], 0) == 1

    def get_perception_mask(self) -> bool:
        return self.regs.get(self.reg_names["PERCEPTION_MASK"], 0) == 1

    def get_rate_factor(self) -> float:
        """Return the CAN rate factor as a float (8.8 fixed -> float)."""
        raw = self.regs.get(self.reg_names["CAN_RATE_FACTOR"], 320)
        return raw / 256.0

    def reset(self):
        """Reset all registers to defaults."""
        for addr, (_, reset, _, _) in REGISTERS.items():
            self.regs[addr] = reset
        self.write_log.clear()

    def get_state(self) -> Dict:
        """Return the full bridge state for the dashboard."""
        regs = {}
        for addr, (name, _, access, desc) in REGISTERS.items():
            regs[name] = {
                "address": f"0x{addr:02X}",
                "value": self.regs.get(addr, 0),
                "access": access,
                "description": desc,
            }
        return {
            "registers": regs,
            "can_periods": self.get_all_can_periods(),
            "gps_enabled": self.get_gps_enabled(),
            "v2x_gain": self.get_v2x_gain(),
            "safe_stop": self.get_safe_stop(),
            "rate_factor": self.get_rate_factor(),
            "write_count": len(self.write_log),
            "recent_writes": self.write_log[-10:] if self.write_log else [],
        }
