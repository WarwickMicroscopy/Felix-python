# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:14:09 2026

@author: Richard
"""

import re
from collections import defaultdict


def parse_orbital_block(text):

    elements = {}
    current_Z = None
    current_name = None

    lines = text.splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # --- Detect new element ---
        header = re.match(r'^([A-Z]+).*?Z\s*=\s*(\d+)', line)
        if header:
            current_name = header.group(1).capitalize()
            current_Z = int(header.group(2))
            # print(f"{current_name}, Z={current_Z}")
            elements[current_Z] = {
                "_name": current_name,
                "s_rows": [],
                "p_rows": defaultdict(list),
                "d_rows": defaultdict(list),
            }
            continue

        tokens = re.split(r'\s+', line)

        # --- split mixed rows ---
        # find second orbital if present
        split_idx = None
        for i, t in enumerate(tokens[1:], 1):
            if re.match(r'\d[spd]', t):
                split_idx = i
                break

        parts = [tokens] if split_idx is None else [tokens[:split_idx],
                                                    tokens[split_idx:]]

        for part in parts:
            orb = part[0]
            # values = list(map(float, part[1:]))
            values = []
            for v in part[1:]:
                try:
                    values.append(float(v))
                except ValueError:
                    continue

            if orb.endswith('s'):
                elements[current_Z]["s_rows"].append((orb, values))

            elif orb.endswith('p'):
                elements[current_Z]["p_rows"][orb].append(values)

            elif orb.endswith('d'):
                elements[current_Z]["d_rows"][orb].append(values)

    # --- Convert to final format ---
    result = {}
    
    for Z, data in elements.items():
    
        entry = {}
        name = data["_name"]
        # print(f"{name}")
    
        # ---- S orbitals ----
        if data["s_rows"]:
            max_cols = max(len(v) for _, v in data["s_rows"])
    
            deltas = [v[0] for _, v in data["s_rows"]]
    
            for i in range(1, max_cols):
                orb = f"{i}s"
                coeffs = []
                n = []
    
                for s, v in data["s_rows"]:
                    coeffs.append(v[i] if i < len(v) else 0.0)
                    n.append(int(s[0]) if i < len(v) else "")
    
                entry[orb] = {
                    "delta": deltas,
                    "coeff": coeffs,
                    "n": n
                }
    
        # ---- P orbitals ----
        for orb, rows in data["p_rows"].items():
            # print(f"{Z}, p orbital {rows[0]}, coeff={rows[1]}")
            entry[orb] = {
                "delta": [r[0] for r in rows],
                "coeff": [r[1] for r in rows],
                "n": [int(orb[0]) for r in rows]
            }
    
        # ---- D orbitals ----
        for orb, rows in data["d_rows"].items():
            entry[orb] = {
                "delta": [r[0] for r in rows],
                "coeff": [r[1] for r in rows],
                "n": [int(orb[0]) for r in rows]
            }
    
        result[Z] = entry

    return result


def print_dict(data, file=None):
    for Z in sorted(data):
        print(f"    {Z}: {{", file=file)
        for orb, vals in data[Z].items():
            print(f"        '{orb}': {{", file=file)
            print(f"            'delta': {vals['delta']},", file=file)
            print(f"            'coeff': {vals['coeff']},", file=file)
            print(f"            'n': {vals['n']}", file=file)
            print("        },", file=file)
        print("    },", file=file)


# %%
with open("Bunge.txt", "rb") as f:
    text = f.read().decode("utf-16", errors="ignore")

data = parse_orbital_block(text)

with open("slater_coefficients.txt", "w") as f:
    print_dict(data, file=f)

