# Simple expert rule engine for battery material recommendations

def recommend_materials(anode, electrolyte, target_v, target_a, temp):
    anode = anode.lower()
    electrolyte = (electrolyte or "").lower()

    # ---------------- VOLTAGE-based rules ----------------
    if target_v >= 4.8:
        cathode = "LNMO (Spinel High Voltage)"
        notes = "Requires high-voltage electrolyte. Add LiBOB or LiDFOB."
    
    elif 4.3 <= target_v < 4.8:
        cathode = "NMC811 (High Nickel)"
        notes = "Good for high voltage and high energy."
    
    elif 3.6 <= target_v < 4.3:
        cathode = "NMC532 or NMC622"
        notes = "Stable mid-voltage chemistry."
    
    else:
        cathode = "LFP (Safe chemistry)"
        notes = "Excellent thermal stability and long cycle life."

    # ---------------- TEMPERATURE rules ----------------
    if temp < 0:
        notes += " Use low-temperature electrolyte (DME-based)."
    elif temp > 50:
        notes += " High-temp detected — use ceramic separator."

    # ---------------- ANODE-based rules ----------------
    if "lto" in anode:
        cathode = "LFP"
        notes += " LTO works best with LFP for extreme safety and long cycle life."
    elif "lithium metal" in anode:
        cathode = "NMC811"
        notes += " Solid-state electrolyte recommended."

    # ---------------- CURRENT rules ----------------
    if target_a >= 6:
        notes += " Use high-porosity electrodes for fast ion transport."

    # ---------------- ELECTROLYTE rules ----------------
    if "fec" in electrolyte:
        notes += " FEC improves SEI on anode, good for fast charge."
    if "lifsi" in electrolyte:
        notes += " LiFSI improves ionic conductivity and thermal stability."

    return {
        "recommended_cathode": cathode,
        "reasoning": notes,
        "inputs": {
            "anode": anode,
            "electrolyte": electrolyte,
            "target_voltage": target_v,
            "target_current": target_a,
            "temperature": temp
        }
    }
