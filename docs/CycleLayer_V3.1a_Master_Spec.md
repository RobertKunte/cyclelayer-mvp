# CycleLayer V3.1a — Master Spec

**BraytonEngine + CycleLayerV3 für N-CMAPSS DS02**

*Single-Source-of-Truth für Claude-Code-Implementierung · Robert Kunte · Mai 2026 · Revision 3.1a*

---

Dieses Dokument ist die verbindliche Spezifikation für die Implementierung des physikalischen Triebwerksmodells (BraytonEngine) und des darauf aufbauenden Hybrid-Prognose-Modells (CycleLayerV3). Es ist die einzige Referenz, die Claude Code für diese Arbeitswelle benutzt.

V3.1a integriert acht implementierungsnahe Patches aus drei externen Review-Runden (Gemini 3.1 und ChatGPT 5.5) plus methodischer Vertiefung zur Health-Parameter-Architektur. Die Patches sind im jeweiligen Abschnitt als "V3.1a Patch Px" markiert. Bestehende Module v1 und V2/PhysResNet bleiben im Repo unangetastet.

---

## Changelog V3.1 → V3.1a (Implementierungs-Patches)

| Nr. | Schwere | Was geändert wurde | Wo |
|-----|---------|--------------------|----|
| **P1** | kritisch | `ETA_INLET` nur im ISA/Ram-Fallback, nicht bei gemessenem N-CMAPSS-P2 | B.2 |
| **P2** | kritisch | `estimate_inlet_flow()` explizit spezifiziert mit Map-Parametrisierung und Bounds | B.3a (neu) |
| **P3** | kritisch | `turbine()` gibt `W_turbine` und `shaft_residual` zurück; Diagnostics um PR_hpt, PR_lpt, P45, P50 erweitert | B.7, B.9 |
| **P4** | kritisch | Health-Modifier-Semantik: θ_phys intern Faktor [0.85,1.00], in Eval als (θ-1) gegen GT-Delta verglichen; AuxHead direkt Delta. Plus Konvertierungstest. | B.8a (neu), D.1, D.5 |
| **P5** | methodisch | `AuxHead.detach()` default für PrognosticsHead-Input, mit YAML-Switch für Ablation | D.1, D.2, D.4 |
| **P6** | methodisch | Sensor-Leakage stärker abgesichert via `mask_target_sensors_prob` im SensorEncoder (Variante B) | C.5a (neu), D.4 |
| **P7** | kosmetisch | Pitch-Sprache präzisiert: zwei post-hoc Pearson + ein supervised diagnostic Pearson | D.5, E.3, E.4 |
| **P8** | technisch | Numerische Safety vor Potenzen in `turbine()`; PR-Clamp-Monitoring als Diagnostics; ETA_DESIGN_HPT/LPT getrennt | B.7, B.9, stations.py |

---

## Changelog V3.0 → V3.1 (Architektur-Korrekturen)

| Nr. | Was geändert wurde | Warum |
|-----|--------------------|-------|
| 1 | LPT-Wellenbilanz im Text korrigiert auf `W_LPT = W_LPC + W_Fan_total` | V3.0 Text in B.6 widersprach dem Pseudocode in B.9. LPT treibt den ganzen Fan, nicht nur den Core-Anteil. |
| 2 | Fan einmaliger Aufruf statt zweimal | Fan ist eine Komponente mit gemeinsamer Welle. Zwei separate Aufrufe sind mechanisch falsch. |
| 3 | θ-Anzahl von 6 auf 5 reduziert (LPT_flow_mod aus BraytonEngine entfernt) | Lokale Flow-Capacity-Effekte erfordern iterativen Solver (V4). Ohne Solver gibt es keine physikalisch saubere Stelle. |
| 4 | Neuer AuxHealthHead für LPT_flow_mod als supervised Diagnose-Target | LPT_flow_mod als 3. Validierungssignal verfügbar, ohne in die Physik einzugreifen. |
| 5 | Sensitivitätstabelle (Stufe C.4) komplett neu — PR/P-basiert statt T-basiert | In Closure mit gemessenen Drehzahlen ist dT durch Wellengleichgewicht fixiert. η wirkt auf PR/P, nicht auf T. |
| 6 | Neues Modul `stations.py` + `units.py` mit Imperial-zu-SI-Konvertierung | N-CMAPSS publiziert in °R, psia, ft, pps, rpm. Ohne expliziten Layer würden Map-Koeffizienten falsche Einheiten kompensieren. |
| 7 | Stufe 3 in 3a/3b/3c gegliedert (healthy-only, degraded, held-out) | V3.0 hätte θ=1.0 gegen vollständig degradierte Sensorverteilung getestet — methodisch falsch. |
| 8 | Loss-Struktur ohne supervidiertes L_theta auf den 5 physikalischen Thetas | θ_phys werden unsupervised getrieben; GT-Korrelation ist post-hoc Evaluationsmetrik. Stärkerer Pitch. |
| 9 | Neuer L_healthy-Term auf early-life Samples (RUL > 80) | Mitigation gegen Identifizierbarkeitsproblem ohne supervidierte θ. |
| 10 | L_sens explizit normalisiert (per-Sensor) | Sonst dominiert P30 (psia, hohe Werte) den Loss gegenüber T-Sensoren. |
| 11 | Neuer Sensor-Leakage-Test in C.5 | Verhindert physikalisch verkleideten Autoencoder — Lehre aus dem V2-Failure. |

---

# Teil A — Architektur und Modulgrenzen

## A.1 Designprinzipien

- **Strikte Modulseparation:** Die thermodynamische Engine (BraytonEngine) ist ein eigenständiges PyTorch-Modul ohne ML-Komponenten. Sie wird zuerst gebaut, vollständig getestet und freigegeben, bevor sie in CycleLayerV3 integriert wird.
- **Erhaltungssätze als harte Constraints:** Massenerhaltung und Wellenleistungsbilanz sind per Konstruktion erfüllt (explicit closure), nicht über Soft-Loss-Terme.
- **Korrigierte Größen als Standardrepräsentation:** Komponenten arbeiten in (m_corr, N_corr) — konsistent mit Walsh & Fletcher, Kurzke, Chao 2022 Appendix A.
- **Vereinfachte parametrische Maps:** Keine OEM-Component-Maps. PR und η werden über 2–3 Polynomkoeffizienten pro Komponente abgebildet.
- **MVP-Vereinfachung Drehzahlen und Fuel:** Nf, Nc und Wf werden als Inputs aus N-CMAPSS-Sensoren gelesen, nicht aus Wellengleichgewicht oder Combustor-Bilanz iterativ gelöst.
- **Vollständig differenzierbar:** Alle Operationen sind PyTorch-native. Keine if-Verzweigungen auf Tensor-Werten, keine numerischen Solver. Bounds via `torch.clamp`.
- **5 Thetas in BraytonEngine — Flow-Capacity in V4:** BraytonEngine bekommt nur die 5 Wirkungsgrad-Modifier. LPT_flow_mod wird in V3.1a als separater AuxHealthHead supervised gelernt, ohne in die Physik einzugreifen.

## A.2 Stationsschema (2-Spool High-Bypass Turbofan)

Die Stationsnummerierung folgt der CMAPSS-/SAE-ARP755-Konvention. Sie wird im Code in dieser Form als Konstanten-Modul abgelegt (`src/cyclelayer/models/stations.py`).

| Station | Beschreibung | Größen | N-CMAPSS Sensor? | Sensor-Spalte |
|---------|--------------|--------|------------------|---------------|
| 0  | Ambient (ISA) | T0, p0, M0 | indirekt (alt, M) | alt, XM |
| 2  | Fan Inlet (Ram) | T2, P2, ṁ_2 | ja | T2, P2 |
| 21 | Fan Outlet Bypass | T21, P21, ṁ_byp | teilweise | P15 |
| **24** | **LPC Outlet** | **T24, P24, ṁ_core** | **ja** | **T24** |
| **30** | **HPC Outlet** | **T30, P30, Ps30, ṁ_core** | **ja** | **T30, P30, Ps30** |
| 4  | Combustor Outlet | T4, P4, ṁ_core+ṁ_f | nein (intern) | — |
| 45 | HPT Outlet | T45, P45 | nein (intern) | — |
| **50** | **LPT Outlet** | **T50, P50** | **ja** | **T50** |

Output-Sensoren der BraytonEngine (vier Stück, fett): T24, T30, P30, T50. P30 ist Total Pressure (Chao 2022 Tab. 2 Spalte 18, C-MAPSS User Guide Tab. 1.2 Index 8). Ps30 (Static) ist als zusätzlicher Sensor verfügbar, wird aber in V3.1a nicht als BraytonEngine-Output verwendet.

## A.3 Datenflussdiagramm (V3.1a mit α+-Architektur)

```
Inputs (aus N-CMAPSS pro Zeitstempel, IMPERIAL):
  ops:    (alt[ft], XM, TRA[%], T2[°R], P2[psia])
  sens:   (Nf[rpm], Nc[rpm], Wf[pps])

       │
       ▼
[ units.py ]   Imperial → SI Konvertierung
       │
       ▼
[ SensorEncoder ]  x_sens (alle 14 Sensoren) → h_sens (B,T,64)
                   [P6: target sensors maskiert mit p=0.5 in training]
[ OpsEncoder ]     ops (5 scen.descriptors)  → z_ops  (B,T,32)
       │
       ├──────────────┬──────────────────┐
       ▼              ▼                  ▼
[ParamHead_phys]  [AuxHealthHead]    (info-only)
   → θ_phys          → lpt_flow_pred
   (B,T,5)           (B,T,1)
   bounded factor    bounded delta
   [0.85, 1.00]      [-0.05, 0.02]

       │ θ_phys
       ▼
[ BraytonEngine ]   ops_SI, Nf, Nc, Wf_SI, θ_phys → ŝ (B,T,4)
                                                   [T24, T30, P30, T50] (SI)
       │
       ▼
[ units.py ]   SI → Imperial für Sensor-Vergleich
       │
       ▼
   ŝ_imperial (B,T,4) ─────► L_sens

       lpt_flow_pred ──► L_aux vs LPT_flow_mod (supervised)

       lpt_flow_pred.detach()   [V3.1a P5 default]
              │
              ▼
[PrognosticsHead]  cat(h_sens, θ_phys, lpt_flow_pred_detached) → RUL (B,T,1)
```

> **Hinweis:** BraytonEngine arbeitet intern in SI. Inputs werden am Eingang konvertiert, Outputs am Ausgang zurück nach Imperial für den Sensor-Loss. `units.py` ist Hard-Gate mit eigenem Test (siehe C.0).

## A.4 Modulgrenzen

Sechs neue Dateien werden angelegt; bestehende Dateien bleiben unverändert.

| Datei | Inhalt |
|-------|--------|
| `src/cyclelayer/models/stations.py` | Stationsnummern, cp/γ-Werte, ISA-Konstanten, Bounds. Pure Konstanten, keine Logik. |
| `src/cyclelayer/models/units.py` | **NEU:** Imperial-zu-SI Konvertierung: ft→m, °R→K, psia→Pa, pps→kg/s. Mit eigenem Test. |
| `src/cyclelayer/models/brayton_engine.py` | BraytonEngine (nn.Module). Eigenständiges Physik-Modul. Keine ML-Komponenten. 5 Thetas. |
| `src/cyclelayer/models/cyclelayer_v3.py` | Hybrid-Modell: Encoder + ParamHead_phys + AuxHealthHead + BraytonEngine + PrognosticsHead. |
| `tests/test_brayton_engine.py` | Validierungs-Suite (Stufen 0–5, siehe Teil C). Hard-Gate für CI. |
| `configs/cyclelayer_v3.yaml` | Trainingskonfiguration für CycleLayerV3. |

> Bestehende Dateien (`brayton_cycle.py`, `cycle_layer.py`, `physresnet.py`, `encoder.py`, `baselines.py`, `prognostics.py`) werden **NICHT** modifiziert. Sie bleiben als historische Referenz und für Vergleichsläufe (CNN-Baseline, v1, V2) verfügbar.

---

# Teil B — BraytonEngine Spezifikation

## B.0 Einheiten-Modul (units.py)

> **V3.1a:** In V3.0 nicht vorhanden. Hard-Gate für die Korrektheit aller Tests in Teil C.

N-CMAPSS publiziert in Imperial, BraytonEngine arbeitet intern in SI. Die Konvertierung ist explizit, dokumentiert und getestet.

```python
# units.py
import torch

# Conversion factors
FT_TO_M    = 0.3048
RANK_TO_K  = 5.0 / 9.0
PSIA_TO_PA = 6894.76
PPS_TO_KGS = 0.4535924      # pounds-mass per second to kg/s
RPM_TO_RAD = 2.0 * 3.14159265 / 60.0

def to_si(ops_imp, sens_imp):
    """Convert N-CMAPSS imperial inputs to SI."""
    return {
        "alt_m":  ops_imp["alt_ft"] * FT_TO_M,
        "mach":   ops_imp["XM"],                   # dimensionless
        "TRA":    ops_imp["TRA_pct"] / 100.0,      # to fraction
        "T2_K":   ops_imp["T2_R"] * RANK_TO_K,
        "P2_Pa":  ops_imp["P2_psia"] * PSIA_TO_PA,
        "Nf_rpm": sens_imp["Nf_rpm"],              # keep rpm for corrected speed
        "Nc_rpm": sens_imp["Nc_rpm"],
        "Wf_kgs": sens_imp["Wf_pps"] * PPS_TO_KGS,
    }

def to_imperial(sensors_si):
    """Convert BraytonEngine outputs back to imperial for sensor-loss."""
    return {
        "T24_R":    sensors_si["T24_K"]  / RANK_TO_K,
        "T30_R":    sensors_si["T30_K"]  / RANK_TO_K,
        "P30_psia": sensors_si["P30_Pa"] / PSIA_TO_PA,
        "T50_R":    sensors_si["T50_K"]  / RANK_TO_K,
    }
```

## B.1 Konstanten (stations.py)

```python
# Thermodynamische Konstanten (SI)
GAMMA_C = 1.40       # heat capacity ratio cold side
GAMMA_T = 1.33       # heat capacity ratio hot side
CP_C    = 1005.0     # J/(kg·K), Air
CP_T    = 1150.0     # J/(kg·K), combustion gas
R_AIR   = 287.05     # J/(kg·K)
LHV     = 43.0e6     # J/kg, Jet-A lower heating value
ETA_COMB = 0.99      # combustor efficiency

# Reference state for corrected quantities (Sea Level Static, ISA)
T_REF = 288.15       # K
P_REF = 101325.0     # Pa

EXP_C = (GAMMA_C - 1.0) / GAMMA_C   # 0.2857
EXP_T = (GAMMA_T - 1.0) / GAMMA_T   # 0.2481

# Soft bounds (für clamp)
ETA_MIN, ETA_MAX = 0.50, 0.99
PR_MIN,  PR_MAX  = 1.05, 25.0

# Inlet
ETA_INLET = 0.98     # pressure recovery (only used in ISA/Ram fallback)

# Bypass
BPR_DESIGN = 5.5     # Standard high-bypass commercial turbofan

# Combustor pressure drop
PI_BURN = 0.04

# Component-specific design efficiencies (V3.1a Patch P8)
ETA_DESIGN_FAN = 0.92
ETA_DESIGN_LPC = 0.90
ETA_DESIGN_HPC = 0.88
ETA_DESIGN_HPT = 0.90
ETA_DESIGN_LPT = 0.92
```

## B.2 Inlet (Station 0 → 2)

> **V3.1a Patch P1:** `ETA_INLET` wird **NUR** im ISA/Ram-Fallback angewendet. Bei gemessenen N-CMAPSS-Werten ist P2 bereits Total Pressure am Fan-Inlet — eine zusätzliche Multiplikation mit 0.98 wäre eine systematische 2%-Verzerrung aller PRs und korrigierten Massenströme.

ISA-Standardatmosphäre + isentropic Ram-Effekte als Fallback. T2 und P2 sind in N-CMAPSS DS02 direkt als Sensoren verfügbar — wenn vorhanden, werden gemessene Werte direkt verwendet.

```python
# In N-CMAPSS DS02 sind T2 und P2 direkt verfügbar als Total-Werte
# am Fan-Inlet, also bereits nach Ram-Recovery.

if use_measured_inlet:
    # Use N-CMAPSS sensors directly; no further inlet correction
    T2 = ops_si["T2_K"]
    P2 = ops_si["P2_Pa"]
else:
    # Fallback: ISA standard atmosphere + Ram + Inlet pressure recovery
    T0 = 288.15 - 0.0065 * ops_si["alt_m"]
    p0 = 101325.0 * (T0 / 288.15) ** 5.2561
    ram_factor_T = 1.0 + 0.5 * (GAMMA_C - 1.0) * ops_si["mach"]**2
    T2 = T0 * ram_factor_T
    P2 = p0 * ram_factor_T ** (GAMMA_C / (GAMMA_C - 1.0))
    P2 = P2 * ETA_INLET   # Inlet pressure recovery only in fallback path
```

## B.3 Korrigierte Größen

```python
def corrected_flow(m_dot, T_in, P_in):
    """Corrected mass flow referenced to Sea Level ISA."""
    return m_dot * torch.sqrt(T_in / T_REF) / (P_in / P_REF)

def corrected_speed(N_rpm, T_in):
    """Corrected rotational speed (rpm)."""
    return N_rpm / torch.sqrt(T_in / T_REF)
```

## B.3a Inlet Mass Flow Estimation (V3.1a Patch P2)

> **V3.1a Patch P2:** In V3.1 war `estimate_inlet_flow()` nur als Aufruf erwähnt, aber nicht spezifiziert. Da `m_in` alle nachfolgenden Werte (W_fan, m_core, W_lpc, W_hpc, T4, T50) skaliert, ist eine explizite, getestete Spezifikation Pflicht. Ohne sie hätte Claude Code freigedreht.

Inlet-Massenstrom wird über eine parametrische Beziehung zu corrected fan speed Nf_corr abgeschätzt. Die Koeffizienten kommen aus der GasTurb-Phase 0 (siehe E.1) und sind initial fixiert.

```python
def estimate_inlet_flow(T2, P2, Nf_rpm, params):
    """
    Estimate total inlet mass flow from corrected fan speed.

    Wc_fan(Nc) = Wc_design * (1 + c1·dN + c2·dN²)
    where dN = (Nc - Nc_design) / Nc_design

    From corrected flow back to actual:
      m_in = Wc_fan · (P2/P_REF) / sqrt(T2/T_REF)
    """
    Nc = corrected_speed(Nf_rpm, T2)
    dN = (Nc - params.Nc_fan_design) / params.Nc_fan_design

    Wc_fan = params.Wc_fan_design * (1.0 + params.c1 * dN
                                          + params.c2 * dN**2)
    Wc_fan = torch.clamp(Wc_fan, params.Wc_min, params.Wc_max)

    m_in = Wc_fan * (P2 / P_REF) / torch.sqrt(T2 / T_REF)
    return m_in

# YAML config (initial values from GasTurb Phase 0):
# brayton_engine:
#   inlet_flow:
#     Wc_fan_design:  ~ 800       # kg/s, CFM56-class SLS Takeoff
#     Nc_fan_design:  ~ 4900      # rpm corrected
#     c1:             ~ 0.85      # linear sensitivity
#     c2:             ~ -0.20     # quadratic curvature
#     Wc_min:         400         # safety bound
#     Wc_max:         1100        # safety bound
```

Validierungstest (gehört in C.1 Stufe 1):

- `m_in` steigt monoton mit `Nc_fan` über realistischen Bereich
- `m_in` liegt in plausiblem Bereich für SLS Takeoff (~500–700 kg/s), Climb, Cruise (~150–300 kg/s)
- Bei `Nc_fan = Nc_design` liefert `m_in ≈ Wc_fan_design · (P2/P_REF) / sqrt(T2/T_REF)`

## B.4 Fan (mit korrigiertem Bypass-Splitting)

> **V3.1:** V3.0-Architektur rief Fan zweimal auf (für Core und Bypass) — mechanisch falsch. V3.1 ruft Fan einmal mit Gesamt-Massenstrom auf, splittet danach.

```python
def fan(T2, P2, m_in_total, Nf_rpm, theta_eta_fan, map_coeffs):
    """
    Single fan call on TOTAL inlet mass flow.
    Returns common (T21, P21) for both core and bypass branches.
    Work is later attributed proportionally to mass flow.
    """
    Wc_fan = corrected_flow(m_in_total, T2, P2)
    Nc_fan = corrected_speed(Nf_rpm, T2)

    PR_fan, eta_fan_nom = parametric_map(Wc_fan, Nc_fan, map_coeffs, "fan")
    eta_fan = eta_fan_nom * theta_eta_fan
    eta_fan = torch.clamp(eta_fan, ETA_MIN, ETA_MAX)

    T21_isen = T2 * PR_fan ** EXP_C
    T21      = T2 + (T21_isen - T2) / eta_fan
    P21      = P2 * PR_fan

    W_fan_total = m_in_total * CP_C * (T21 - T2)
    return T21, P21, W_fan_total, eta_fan, PR_fan

# Bypass split AFTER fan
m_byp  = m_in_total * BPR_DESIGN / (BPR_DESIGN + 1.0)
m_core = m_in_total / (BPR_DESIGN + 1.0)

# Work attribution proportional to mass flow
W_fan_byp  = W_fan_total * m_byp  / m_in_total
W_fan_core = W_fan_total * m_core / m_in_total
```

## B.5 LPC, HPC (Core-Verdichter)

```python
def compressor(T_in, P_in, m_dot, N_rpm, theta_eta, map_coeffs, design_PR):
    """
    Generic axial compressor with parametric map.
    Used for LPC (N=Nf) and HPC (N=Nc).
    """
    Wc = corrected_flow(m_dot, T_in, P_in)
    Nc = corrected_speed(N_rpm, T_in)

    PR, eta_nom = parametric_map(Wc, Nc, map_coeffs, kind="compressor",
                                 design_PR=design_PR)
    eta = torch.clamp(eta_nom * theta_eta, ETA_MIN, ETA_MAX)
    PR  = torch.clamp(PR, PR_MIN, PR_MAX)

    T_out_isen = T_in * PR ** EXP_C
    T_out      = T_in + (T_out_isen - T_in) / eta
    P_out      = P_in * PR
    W_comp     = m_dot * CP_C * (T_out - T_in)

    return T_out, P_out, W_comp, eta, PR

# Note: design_PR seeded from GasTurb generic turbofan:
#   Fan ≈ 1.6, LPC ≈ 2.0, HPC ≈ 12.0
# Map coefficients from Kurzke generic maps (initial fixed)
```

## B.6 Combustor (Station 30 → 4)

```python
def combustor(T30, P30, m_core, Wf):
    """
    Energy balance: m_core·cp_c·T30 + Wf·LHV·η_comb = (m_core + Wf)·cp_t·T4
    Pressure drop:  P4 = P30·(1 - π_b)
    """
    m_4 = m_core + Wf
    T4  = (m_core * CP_C * T30 + Wf * LHV * ETA_COMB) / (m_4 * CP_T)
    P4  = P30 * (1.0 - PI_BURN)
    return T4, P4, m_4
```

## B.7 Turbinen mit Explicit Closure

> **V3.1a Patches P3 + P8:** `turbine()` liefert jetzt `W_turbine` und `shaft_residual` zurück (sonst lassen sich die Closure-Tests aus C.1 nicht implementieren). Plus numerische Safety vor der Potenz-Operation. Plus ETA_DESIGN getrennt für HPT und LPT.

Turbinen-PR ergibt sich aus der Wellenleistungsbilanz, nicht aus einem Map. Energiebilanz wird per Konstruktion erfüllt, nicht per Soft-Loss.

```python
def turbine(T_in, P_in, m_dot, W_required, theta_eta, eta_design):
    """
    Inverse problem: given required shaft power, find PR such that
        m_dot · cp_t · ΔT = W_required
    Returns (T_out, P_out, W_turbine, shaft_residual, eta, PR).

    eta_design is component-specific:
       HPT: ETA_DESIGN_HPT = 0.90
       LPT: ETA_DESIGN_LPT = 0.92
    """
    eta = torch.clamp(theta_eta * eta_design, ETA_MIN, ETA_MAX)

    # Actual temperature drop from energy balance
    dT     = W_required / (m_dot * CP_T)
    T_out  = T_in - dT

    # Isentropic temperature drop (η = ΔT_actual / ΔT_isen)
    dT_isen      = dT / eta
    T_out_isen   = T_in - dT_isen

    # Numerical safety BEFORE the power operation (V3.1a Patch P8):
    # Without these clamps, random batches in the gradient-stability
    # test produce NaN before reaching the PR clamp.
    eps = 1e-6
    T_out_isen = torch.clamp(T_out_isen, min=0.05 * T_in)
    ratio_in   = torch.clamp(T_in / T_out_isen,
                             min=1.0 + eps,
                             max=PR_MAX ** EXP_T)

    PR    = ratio_in ** (1.0 / EXP_T)
    PR    = torch.clamp(PR, PR_MIN, PR_MAX)
    P_out = P_in / PR

    # Compute actual extracted work and shaft residual for closure check
    W_turbine      = m_dot * CP_T * (T_in - T_out)
    shaft_residual = W_turbine - W_required   # should be ~0 by construction

    return T_out, P_out, W_turbine, shaft_residual, eta, PR
```

**Wellenbilanzen (V3.1a):**

```
HPT-Welle:  W_HPT = W_HPC
LPT-Welle:  W_LPT = W_LPC + W_Fan_total
            = W_LPC + W_Fan_core + W_Fan_byp
```

## B.8 θ-Parameter (5 Thetas in BraytonEngine)

> **V3.1:** In V3.0 waren 6 Thetas inkl. θ_m_hpt definiert. V3.1 reduziert auf 5, weil lokale Flow-Capacity ohne iterativen Solver nicht physikalisch sauber abbildbar ist. LPT_flow_mod wird stattdessen als AuxHealthHead supervised gelernt (siehe Teil D).

| Symbol | Wirkung | Bound (training) | N-CMAPSS GT |
|--------|---------|------------------|-------------|
| `θ_η_fan` | Fan-Wirkungsgrad | [0.85, 1.00] | nicht in DS02 |
| `θ_η_lpc` | LPC-Wirkungsgrad | [0.85, 1.00] | nicht in DS02 |
| `θ_η_hpc` | HPC-Wirkungsgrad | [0.85, 1.00] | nicht in DS02 |
| **`θ_η_hpt`** | **HPT-Wirkungsgrad** | **[0.85, 1.00]** | **HPT_eff_mod** |
| **`θ_η_lpt`** | **LPT-Wirkungsgrad** | **[0.85, 1.00]** | **LPT_eff_mod** |

Die zwei fett markierten θ-Parameter (η_hpt, η_lpt) korrespondieren direkt mit N-CMAPSS-Ground-Truth. Sie werden im Training **NICHT** supervidiert — Vergleich mit GT erfolgt post-hoc als Pearson-Korrelation in der Evaluation.

## B.8a Health-Modifier-Semantik: Faktor vs. Delta (V3.1a Patch P4)

> **V3.1a Patch P4:** V3.1 hatte θ_phys auf [0.85, 1.00] (Faktor) und AuxHead auf [-0.05, 0.02] (Delta). Diese Inkonsistenz musste sauber aufgelöst werden, weil sonst die GT-Vergleiche auf zwei Skalen liefen und Magnitude-Behauptungen im Pitch falsch wären.

In Chao 2022 Tab. 4 und Fig. 6 sind die N-CMAPSS-Health-Parameter HPT_eff_mod, LPT_eff_mod, LPT_flow_mod als Delta um 0 publiziert (gesund = 0, degradiert = negativ um typisch -0.025). V3.1a regelt die Semantik wie folgt eindeutig:

| Größe | Repräsentation intern | N-CMAPSS GT-Format | Eval-Konvertierung |
|-------|----------------------|--------------------|--------------------|
| θ_η_hpt, θ_η_lpt (alle 5 θ_phys) | Faktor [0.85, 1.00] | Delta um 0 | (θ_phys − 1.0) gegen GT |
| `lpt_flow_pred` (AuxHead) | Delta [-0.05, 0.02] | Delta um 0 | direkt gegen GT |

Implementierungs-Helfer und Konvertierungstest:

```python
def theta_phys_as_delta(theta_phys):
    """Convert internal factor representation to delta for GT comparison."""
    return theta_phys - 1.0

def theta_phys_from_factor(factor):
    """Identity (clarity wrapper)."""
    return factor

# Pflicht-Test: early-life median ist nahe der gesunden Referenz
# - median(theta_phys at RUL > 80, axis=units) ≈ 1.00 ± 0.01
# - median(theta_phys_as_delta at RUL > 80) ≈ 0.00 ± 0.01
# - median(lpt_flow_pred at RUL > 80) ≈ 0.00 ± 0.005
```

## B.9 BraytonEngine.forward() — Pseudocode (V3.1a)

> **V3.1a Patches P3 + P5 + P8:** `turbine()` liefert jetzt sechs Werte zurück inkl. `W_turbine` und `shaft_residual`. Diagnostics um PR_hpt, PR_lpt, P45, P50 sowie PR-Clamp-Monitoring erweitert.

```python
class BraytonEngine(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.register_buffer("map_fan", torch.tensor(config.map_fan))
        self.register_buffer("map_lpc", torch.tensor(config.map_lpc))
        self.register_buffer("map_hpc", torch.tensor(config.map_hpc))
        self.inlet_flow_params = config.inlet_flow         # V3.1a P2
        self.bpr_design        = BPR_DESIGN
        self.eta_design_hpt    = config.eta_design_hpt     # 0.90
        self.eta_design_lpt    = config.eta_design_lpt     # 0.92

    def forward(self, ops_si, sens_si, theta_phys):
        """
        Args:
            ops_si:     dict with T2_K, P2_Pa, alt_m, mach, TRA
            sens_si:    dict with Nf_rpm, Nc_rpm, Wf_kgs
            theta_phys: (B, T, 5) — eta_fan, eta_lpc, eta_hpc, eta_hpt, eta_lpt
                        all bounded factor [0.85, 1.00], 1.00 = healthy

        Returns:
            sensors_pred_si: dict T24_K, T30_K, P30_Pa, T50_K
            diagnostics:     dict for tests/logging (extended in V3.1a)
        """
        T2 = ops_si["T2_K"]
        P2 = ops_si["P2_Pa"]   # already total at fan inlet (V3.1a P1)
        Nf, Nc, Wf = sens_si["Nf_rpm"], sens_si["Nc_rpm"], sens_si["Wf_kgs"]

        # Inlet flow (V3.1a P2: explicit spec, see B.3a)
        m_in = estimate_inlet_flow(T2, P2, Nf, self.inlet_flow_params)

        # Fan (single call on total mass flow)
        T21, P21, W_fan_total, eta_fan, PR_fan = self._fan(
            T2, P2, m_in, Nf, theta_phys[..., 0])

        # Bypass split (after fan)
        m_byp  = m_in * self.bpr_design / (self.bpr_design + 1.0)
        m_core = m_in - m_byp
        W_fan_core = W_fan_total * m_core / m_in
        W_fan_byp  = W_fan_total * m_byp  / m_in

        # Core compressors
        T24, P24, W_lpc, eta_lpc, PR_lpc = self._lpc(
            T21, P21, m_core, Nf, theta_phys[..., 1])
        T30, P30, W_hpc, eta_hpc, PR_hpc = self._hpc(
            T24, P24, m_core, Nc, theta_phys[..., 2])

        # Combustor
        T4, P4, m_4 = self._combustor(T30, P30, m_core, Wf)

        # Turbines (closure):
        # HPT-Welle:  W_HPT = W_HPC
        T45, P45, W_hpt, hpt_residual, eta_hpt, PR_hpt = self._hpt(
            T4, P4, m_4,
            W_required=W_hpc,
            theta_eta=theta_phys[..., 3],
            eta_design=self.eta_design_hpt)

        # LPT-Welle:  W_LPT = W_LPC + W_Fan_total
        W_lpt_required = W_lpc + W_fan_total
        T50, P50, W_lpt, lpt_residual, eta_lpt, PR_lpt = self._lpt(
            T45, P45, m_4,
            W_required=W_lpt_required,
            theta_eta=theta_phys[..., 4],
            eta_design=self.eta_design_lpt)

        sensors_pred_si = {
            "T24_K": T24, "T30_K": T30, "P30_Pa": P30, "T50_K": T50
        }

        # Extended diagnostics (V3.1a P3 + P8)
        diagnostics = {
            # Mass and energy balance residuals
            "mass_balance_inlet":   (m_in - (m_byp + m_core)).abs(),
            "mass_balance_combust": (m_4  - (m_core + Wf)).abs(),
            "shaft_HPT_residual":   hpt_residual,
            "shaft_LPT_residual":   lpt_residual,

            # Component work (for tests + plots)
            "W_fan_total": W_fan_total, "W_fan_core": W_fan_core,
            "W_fan_byp":   W_fan_byp,
            "W_lpc": W_lpc, "W_hpc": W_hpc,
            "W_hpt": W_hpt, "W_lpt": W_lpt,

            # Pressure ratios (needed for sensitivity tests, V3.1a P3)
            "PR_fan": PR_fan, "PR_lpc": PR_lpc, "PR_hpc": PR_hpc,
            "PR_hpt": PR_hpt, "PR_lpt": PR_lpt,

            # Internal stations (for sensitivity tests)
            "T4": T4, "P4": P4, "m_4": m_4,
            "T45": T45, "P45": P45,
            "P50": P50,

            # Effective efficiencies after theta and clamp
            "eta_fan": eta_fan, "eta_lpc": eta_lpc, "eta_hpc": eta_hpc,
            "eta_hpt": eta_hpt, "eta_lpt": eta_lpt,

            # Overall metrics
            "P30_over_P2": P30 / P2,

            # PR-Clamp monitoring (V3.1a P8) — fraction of samples at clamp
            "frac_PR_fan_clamped": ((PR_fan == PR_MIN) | (PR_fan == PR_MAX)).float().mean(),
            "frac_PR_lpc_clamped": ((PR_lpc == PR_MIN) | (PR_lpc == PR_MAX)).float().mean(),
            "frac_PR_hpc_clamped": ((PR_hpc == PR_MIN) | (PR_hpc == PR_MAX)).float().mean(),
            "frac_PR_hpt_clamped": ((PR_hpt == PR_MIN) | (PR_hpt == PR_MAX)).float().mean(),
            "frac_PR_lpt_clamped": ((PR_lpt == PR_MIN) | (PR_lpt == PR_MAX)).float().mean(),
        }
        return sensors_pred_si, diagnostics
```

---

# Teil C — Validierungssuite

Sechs Stufen, Stufen 0–5 als pytest in CI, Stufe 6 (GasTurb-Cross-Check) offline einmalig.

## C.0 Stufe 0 — Einheiten-Konvertierung

Hard-Gate vor allen anderen Tests. Wenn `units.py` falsch ist, sind alle anderen Tests bedeutungslos.

- Roundtrip: imperial → SI → imperial liefert Ausgangswerte (relative Abweichung < 1e-6)
- Bekannte Referenzwerte: 100 °R = 55.556 K, 14.7 psia ≈ 101353 Pa, 1000 ft = 304.8 m
- ISA Sea Level: T_ref = 518.67 °R = 288.15 K, P_ref = 14.696 psia = 101325 Pa

## C.1 Stufe 1 — Erhaltungssätze (hart)

- Massenerhaltung Inlet: `|ṁ_in − (ṁ_byp + ṁ_core)| / ṁ_in < 1e-6`
- Massenerhaltung Combustor: `|ṁ_4 − (ṁ_core + ṁ_f)| / ṁ_4 < 1e-6`
- HPT-Welle: `|W_HPT − W_HPC| / W_HPC < 1e-4`
- LPT-Welle: `|W_LPT − (W_LPC + W_Fan_total)| / W_LPT < 1e-4`
- Combustor-Energiebilanz: `|Δh_actual − ṁ_f · LHV · η_comb| / Energy_in < 1e-3`

## C.2 Stufe 2 — Physikalische Plausibilität (weich)

| Größe | Erwartung | Quelle |
|-------|-----------|--------|
| Fan PR | 1.4 – 1.7 | Walsh & Fletcher Tab. 5.3 |
| LPC PR | 1.5 – 2.5 | Walsh & Fletcher |
| HPC PR | 8 – 16 | CMAPSS-Klasse |
| Overall PR (P30/P2) | 20 – 40 | Standard Turbofan |
| BPR (configured) | ≈ 5.5 | CFM56-Klasse |
| T4 (TIT) | 1300 – 1900 K | Materialgrenze |
| T-Monotonie 2→4 | T2 < T24 < T30 < T4 | Brayton-Topologie |
| T-Monotonie 4→50 | T4 > T45 > T50 | Brayton-Topologie |
| alle η | 0.7 – 0.99 | Realistic Range |

## C.3 Stufe 3 — N-CMAPSS Range Match (gestaffelt)

> **V3.1:** V3.0 hätte θ=1.0-Modell gegen vollständig degradierte Sensorverteilung getestet. V3.1 staffelt nach Lebenszustand.

### Stufe 3a — Healthy-only

Test mit θ=1.0 gegen Samples mit RUL > 80 (gesunde Frühphase). Für jeden der 4 Output-Sensoren wird verlangt: predicted distribution überlappt mit measured distribution mit IoU > 0.7 (Quantile 5–95%). Toleranz strenger als V3.0, weil hier kein Degradationseffekt erwartet wird.

### Stufe 3b — Degraded mit GT-Thetas

Test mit θ_η_hpt und θ_η_lpt aus N-CMAPSS-Ground-Truth gesetzt (HPT_eff_mod, LPT_eff_mod als 1+mod), andere θ=1.0. Samples mit RUL < 30. IoU > 0.6.

### Stufe 3c — Held-out Units

Map-Koeffizienten dürfen nur auf Train-Units kalibriert werden. Held-out Test-Units müssen Stufe 3a/3b ohne Re-Kalibrierung bestehen.

## C.4 Stufe 4 — Sensitivitäts-Korrektheit

> **Warnung — V3.0-Tabelle behauptete:** "θ_η_hpt sinkt → T50 steigt stark". Das ist in unserer Closure FALSCH, weil dT durch das Wellengleichgewicht und gemessene Drehzahlen fixiert ist. η wirkt hier auf PR/P, nicht auf T.

In der V3.1-Closure mit gemessenen Nf/Nc und Wf ist die Wellenleistung W_required festgelegt durch die Kompressor-Bilanz. Daraus folgt: dT_lpt = W_required/(m·cp) ist fixiert, und damit auch T_out. η wirkt nur auf das resultierende isentropische Temperaturverhältnis und damit auf das Druckverhältnis PR.

| θ-Parameter | Variation | Erwartete Sensor-Antwort | Mathematischer Grund |
|-------------|-----------|--------------------------|----------------------|
| `θ_η_fan` ↓ 5% | 0.95 | T24 ↑, T30 ↑, P30: indirekt über Map(Wc,Nc), T50 ↑ | Fan η ↓ → T21 ↑ → kaskadiert nach T24, T30, T50 (Energiezunahme) |
| `θ_η_hpc` ↓ 5% | 0.95 | T30 ↑, P30 ≈, T50 ↑ | HPC η ↓ → T30 ↑ → W_HPC ↑ → W_HPT ↑ → ΔT_HPT ↑ |
| **`θ_η_hpt` ↓ 5%** | **0.95** | **T45 ≈ unverändert, P45 ↓ (PR_hpt ↑)** | **dT_hpt = W_HPC/(m·cp) fixed by shaft balance. η ↓ → dT_isen ↑ → PR ↑ → P out ↓** |
| **`θ_η_lpt` ↓ 5%** | **0.95** | **T50 ≈ unverändert, P50 ↓ (PR_lpt ↑)** | **Analog: dT_lpt fixed by W_LPC + W_Fan_total. η ↓ → PR ↑ → P50 ↓** |
| `θ_η_lpc` ↓ 5% | 0.95 | T24 ↑, indirekte Effekte stromabwärts | LPC η ↓ → T24 ↑ → kaskadiert |

**Zentrale Beobachtung der Mathematik** (gilt nur in der explicit-closure-Architektur mit gemessenen Drehzahlen): Turbinen-η wirkt primär auf Drücke, nicht auf Temperaturen. Dies ist methodisch wichtig zu kommunizieren — die Sensitivitätstests prüfen diese Mathematik, nicht ein "echtes" Triebwerksverhalten unter freier Drehzahl-Anpassung.

## C.5 Stufe 5 — Gradient-Stabilität und Sensor-Leakage

- Forward Pass: keine NaN, keine Inf bei beliebigen Inputs
- Backward Pass: finite Gradienten an allen 5 θ-Inputs
- Gradient-Norm bleibt unter 1e6 für realistische θ-Range
- Wiederholte Forward/Backward-Calls (50 random batches) — keine NaN-Episode
- `torch.autograd.gradcheck` mit double precision für eine reduzierte Modellinstanz
- **PR-Clamp-Aktivität (V3.1a P8):** Healthy-Range < 5% Samples am Clamp pro Komponente; degraded Range < 10%. Wenn höher: Map-Koeffizienten oder Bounds revidieren.

### C.5a Sensor-Leakage-Tests (V3.1a Patch P6 — Variante B)

> **V3.1a P6:** Der V3.1-Random-Input-Test deckt das Hauptleck nicht ab. Wenn der SensorEncoder die Target-Sensoren T24/T30/P30/T50 sieht und die BraytonEngine sie wieder rekonstruieren soll, kann ein physikalisch verkleideter Autoencoder entstehen — genau das V2-Problem.

V3.1a setzt zwei komplementäre Tests ein, plus eine architektonische Maßnahme im SensorEncoder selbst:

- **Architektonische Maßnahme:** Im SensorEncoder werden die vier Target-Sensoren während Training mit Wahrscheinlichkeit p (default 0.5) auf 0 maskiert, bevor der Encoder sie sieht. Pflicht-YAML-Parameter: `encoder.mask_target_sensors_prob: 0.5`. Bei Inferenz keine Maskierung.
- **Test 1 — Random-Input-Vergleich:** Trainiere für 1 Epoche mit randomisierten Inputs. RMSE muss signifikant schlechter sein (>2x) als mit echten Inputs.
- **Test 2 — Target-Sensor-Maskierung:** Bei Inferenz alle Target-Sensoren (T24, T30, P30, T50) auf 0 setzen und nur restliche Sensoren als Input geben. Modell muss noch immer plausible θ_phys liefern (Pearson mit GT > 0.5; Schwelle weicher als 0.7, weil Information reduziert).
- **Test 3 (Ablation, in Phase F):** Vergleich V3-mit-Maskierung vs. V3-ohne-Maskierung auf Test-Set. Ohne-Maskierung sollte nicht dramatisch besser sein. Wenn dramatisch besser: Sensor-Leak ist real und Pitch muss angepasst werden.

## C.6 Stufe 6 — GasTurb Cross-Check (offline, manuell)

Externe Plausibilisierung gegen ein etabliertes Tool. Wird einmal durchgeführt, nicht in CI.

- In GasTurb: generischer 2-Spool High-Bypass Turbofan, BPR ≈ 5.5, CFM56-Klasse
- Drei Operating Points: SLS Takeoff, Climb (alt 25 kft, M 0.7), Cruise (alt 35 kft, M 0.8)
- In beiden Modellen: vier Output-Sensoren bei θ = 1.0 vergleichen
- Akzeptanz: relative Abweichung pro Sensor < 15% an allen drei Punkten

Bei Abweichung > 15%: Map-Koeffizienten korrigieren, dokumentieren, neu validieren.

---

# Teil D — Integration in CycleLayerV3 (α+ Architektur)

## D.1 Hybrid-Modell-Architektur

> **V3.1a Patches P5 + P7:** AuxHead-Output wird default mit `detach()` in den PrognosticsHead gegeben. Pitch-Sprache präzisiert: zwei post-hoc Pearson + ein supervised diagnostic Pearson.

```
Inputs:
  x_sens (B,T,14)   — alle N-CMAPSS Sensoren (für Encoder)
                      [V3.1a P6: target sensors mit p maskiert in training]
  ops    (B,T,5)    — alt, mach, TRA, T2, P2
  Nf, Nc, Wf        — Subset von x_sens, an BraytonEngine

       │
       ▼
[ SensorEncoder ]    x_sens → h_sens (B,T,64)
[ OpsEncoder ]       ops    → z_ops  (B,T,32)

       │
       ├──────────────┬──────────────────┐
       ▼              ▼                  ▼
[ParamHead_phys]  [AuxHealthHead]    (parallel)
  Linear+sigmoid    Linear+tanh-scale
  → θ_phys (B,T,5)  → lpt_flow_pred (B,T,1)
  bounded factor    bounded delta
  [0.85, 1.00]      [-0.05, 0.02]

       │ θ_phys
       ▼
[ BraytonEngine ]    ops_SI, Nf, Nc, Wf_SI, θ_phys → ŝ_SI (B,T,4)
       │
       ▼
   ŝ → L_sens (normalized) vs measured T24/T30/P30/T50

       lpt_flow_pred ──► L_aux vs LPT_flow_mod (supervised)

       lpt_flow_pred.detach()  [V3.1a P5 default]
              │
              ▼
[ PrognosticsHead ]  cat(h_sens, θ_phys, lpt_flow_pred_detached)
                     → RUL (B,T,1)
```

## D.2 AuxHealthHead — Spezifikation

Separater Head, der LPT_flow_mod als supervised Diagnose-Target lernt. Wirkt **NICHT** in die BraytonEngine ein. Begründung: lokale Flow-Capacity ohne iterativen Solver lässt sich nicht physikalisch sauber abbilden — siehe V4-Roadmap.

```python
class AuxHealthHead(nn.Module):
    """Predicts LPT_flow_mod as auxiliary diagnostic target.
    Output bounded to typical N-CMAPSS range [-0.05, 0.02] (delta)."""
    def __init__(self, in_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.lo, self.hi = -0.05, 0.02

    def forward(self, h):
        raw = self.net(h)
        return self.lo + (self.hi - self.lo) * (torch.tanh(raw) + 1.0) / 2.0


# ============================================================
# Aufruf-Stelle in CycleLayerV3 (V3.1a Patch P5)
# ============================================================
# AuxHead wird default mit detach() in den PrognosticsHead gegeben,
# damit RUL-Loss den AuxHead nicht in eine RUL-optimale (statt
# LPT_flow_mod-optimale) Richtung verbiegen kann.

lpt_flow_pred = self.aux_health_head(h_sens)   # (B,T,1)

# L_aux uses lpt_flow_pred WITH grad
L_aux = mse_loss(lpt_flow_pred, lpt_flow_mod_true)

# PrognosticsHead receives DETACHED feature by default
if config.aux_head.detach_for_rul:
    lpt_feature = lpt_flow_pred.detach()
else:
    lpt_feature = lpt_flow_pred

rul_pred = self.prognostics_head(
    torch.cat([h_sens, theta_phys, lpt_feature], dim=-1))
```

## D.3 Loss-Struktur

```
L_total = λ_rul     · L_rul
        + λ_sens    · L_sens_normalized
        + λ_aux     · L_aux              # supervised, klein
        + λ_healthy · L_healthy          # early-life prior
        + λ_smooth  · L_smooth

# Komponenten:
L_rul     = asymmetric_RUL_loss(rul_pred, rul_true)
L_sens    = mean over (T24, T30, P30, T50) of:
              MSE((ŝ_imperial - measured) / sigma_per_sensor)
L_aux     = MSE(lpt_flow_pred, LPT_flow_mod_true)
L_healthy = mean over RUL > 80:  ||θ_phys - 1.0||²
L_smooth  = mean( (θ_phys[:,1:] - θ_phys[:,:-1])² )
          + mean( (lpt_flow_pred[:,1:] - lpt_flow_pred[:,:-1])² )

# Initial-Gewichte:
λ_rul     = 1.0
λ_sens    = 0.5
λ_aux     = 0.05    # bewusst klein
λ_healthy = 0.10
λ_smooth  = 1e-3
```

> **Wichtig:** KEIN supervised L_theta auf θ_phys. Die 5 physikalischen Thetas werden ausschließlich durch L_sens und L_rul getrieben. GT-Korrelation für HPT_eff_mod und LPT_eff_mod ist post-hoc Evaluationsmetrik. λ_aux bleibt klein, damit AuxHead nicht den SensorEncoder dominiert.

## D.4 Trainings-Konfiguration (`cyclelayer_v3.yaml`)

```yaml
model:
  type: cyclelayer_v3
  encoder:
    type: cnn1d
    channels: [32, 64, 64]
    kernel: 5
    mask_target_sensors_prob: 0.5    # V3.1a P6: mask T24/T30/P30/T50 in training
  param_head_phys:
    hidden: [64, 32]
    theta_dim: 5
    representation: factor            # V3.1a P4: internal factor [0.85, 1.00]
    bounds: [0.85, 1.00]
  aux_health_head:
    hidden: [32]
    representation: delta             # V3.1a P4: explicit delta around 0
    output_bounds: [-0.05, 0.02]
    detach_for_rul: true              # V3.1a P5: default detach for prognostics input
  prognostics_head:
    hidden: [64, 32]
  brayton_engine:
    use_measured_inlet: true          # V3.1a P1: skip ETA_INLET on measured P2
    bpr_design: 5.5
    eta_design_hpt: 0.90              # V3.1a P8
    eta_design_lpt: 0.92              # V3.1a P8
    inlet_flow:                       # V3.1a P2: explicit inlet flow estimator
      Wc_fan_design: 800.0
      Nc_fan_design: 4900.0
      c1: 0.85
      c2: -0.20
      Wc_min: 400.0
      Wc_max: 1100.0

data:
  use_ops: true
  use_lpt_flow_truth: true
  window_size: 50
  stride: 1
  health_modifier_format: delta       # V3.1a P4: how N-CMAPSS GT is stored

training:
  lr: 1e-4
  batch_size: 256
  max_epochs: 60
  grad_clip_norm: 0.5
  amp: true
  early_stopping_patience: 10
  loss_weights:
    rul:     1.0
    sens:    0.5
    aux:     0.05
    healthy: 0.10
    smooth:  1e-3
  healthy_rul_threshold: 80
```

## D.5 Akzeptanzkriterien Benchmark v1

Diese Metriken bilden das Benchmark-v1-Deliverable. Verglichen wird gegen v1, V2/PhysResNet und CNN-Baseline.

- `test/RMSE_cycle ≤ 9.0` (CNN-Baseline ≈ 10.7)
- `test/S-score ≤ 1.10` (CNN-Baseline ≈ 1.13)
- **Pearson(θ_η_hpt, HPT_eff_mod) > 0.7** — *post-hoc, primary moat metric*
- **Pearson(θ_η_lpt, LPT_eff_mod) > 0.7** — *post-hoc, primary moat metric*
- Pearson(lpt_flow_pred, LPT_flow_mod) > 0.7 — *supervised diagnostic*
- `PH_median` berechenbar für ≥ 80% der Test-Units
- 20%-Daten-Ablation: V3-RMSE_cycle bei 20% Trainingsdaten degradiert um < 20%

---

# Teil E — Phasenplan und CLAUDE.md-Anweisungen

## E.1 Phasenstruktur (V3.1a mit GasTurb-Stunde als Phase 0)

| Phase | Inhalt | Deliverable | Akzeptanz |
|-------|--------|-------------|-----------|
| **0 — Stunde 1** | **GasTurb-Konfiguration** | **CFM56-Klasse Standard-Turbofan, 3 OPs, Map-Koeffizienten extrahiert** | **Initial map_coeffs in YAML eingetragen** |
| A — Tag 1–2 | `units.py` + `stations.py` + BraytonEngine Skeleton | Skelett-Forward läuft, Output-Shape korrekt | Stufe 0 grün |
| B — Tag 3–4 | Erhaltungs- und Plausibilitäts-Tests | Alle Stufe-1 + Stufe-2 + Stufe-5 Tests grün | pytest grün, Conservation < tol |
| C — Tag 5 | Map-Tuning auf N-CMAPSS DS02 | Stufe 3a/3b/3c grün | IoU-Match auf early/degraded/heldout |
| D — Tag 6 | Sensitivität + GasTurb-Cross-Check | Stufe 4 + Stufe 6 grün | Vorzeichen korrekt, GasTurb-Diff <15% |
| **E — Tag 7–9** | **CycleLayerV3 Integration (α+)** | **`cyclelayer_v3.py` + AuxHead + config + Trainer-Dispatch** | **1 Epoche ohne NaN, Sensor-Leakage-Test grün** |
| **F — Tag 10–12** | **Volltraining + 4-Wege-Vergleich** | **Notebook v1 + V2 + CNN + V3** | **V3 ≤ CNN auf RMSE_cycle und PH** |
| **G — Tag 13–14** | **20%-Ablation + θ-Plots + Pearson-Korrelationen** | **Benchmark-v1 One-Pager** | **2× post-hoc + 1× supervised Pearson > 0.7** |

## E.2 CLAUDE.md-Ergänzung

```
## Aktueller Fokus: CycleLayer V3.1a / BraytonEngine

Verbindliche Spec: docs/CycleLayer_V3.1a_Master_Spec.md (Mai 2026, Rev 3.1a).
ALLE Implementierungsentscheidungen folgen diesem Dokument.

### Hard Rules
1. units.py + stations.py + brayton_engine.py werden ZUERST gebaut
   und VOLLSTÄNDIG getestet bevor cyclelayer_v3.py angefasst wird.
2. Bestehende Module (brayton_cycle.py, cycle_layer.py, physresnet.py,
   v1, baselines.py, prognostics.py, encoder.py) werden NICHT modifiziert.
3. Erhaltungssätze sind harte Constraints (closure), KEINE soft losses.
4. Drehzahlen Nf, Nc und Fuel Wf kommen aus N-CMAPSS-Sensoren
   (nicht aus Solver, MVP-Vereinfachung).
5. Map-Koeffizienten und inlet_flow Parameter kommen aus GasTurb (Phase 0)
   und sind initial fixiert.
6. Vor jeder Phase E (Integration) müssen Stufen 0,1,2,3,4,5,5a grün sein.
7. tests/test_brayton_engine.py ist Hard-Gate für CI.
8. BraytonEngine bekommt 5 Thetas (alle Wirkungsgrade, Faktor-Repräsentation).
9. LPT_flow_mod NIEMALS in BraytonEngine einbauen — nur als AuxHealthHead.
10. KEIN supervised L_theta auf θ_phys. Diese werden unsupervised gelernt.
11. ETA_INLET nur im ISA/Ram-Fallback, NIE auf gemessenes P2 anwenden (P1).
12. estimate_inlet_flow() nutzt explizite YAML-Parameter, keine
    Heuristik-Defaults im Code (P2).
13. turbine() liefert (T_out, P_out, W_turbine, shaft_residual, eta, PR).
    shaft_residual muss in Diagnostics — sonst Tests nicht implementierbar (P3).
14. AuxHead-Output mit detach() in PrognosticsHead. Default true (P5).
15. SensorEncoder maskiert Target-Sensoren während Training mit p=0.5 (P6).
16. θ_phys ist Faktor [0.85,1.00] intern; Vergleich mit GT-Delta via (θ-1) (P4).

### Out of Scope (NICHT bauen)
- OEM-Component-Maps oder Lookup-Tables
- Iterative Solver für Wellengleichgewichte oder Choked-Flow
- Lokale Flow-Capacity-Modellierung in BraytonEngine (V4-Material)
- Bleed Air, Variable Geometry, Cooling Flows
- Reynolds-Korrekturen, Mach-Effekte über Inlet-Recovery hinaus
- Multi-Stage Compressor/Turbine Splits

### Wenn etwas unklar ist
Frag NICHT Claude Code-intern, sondern stoppe und melde dich bei Robert.
Insbesondere bei: Map-Koeffizienten, inlet_flow-Parametern, Toleranzen,
Loss-Gewichtungen, oder wenn du eine "elegantere" Lösung sehen würdest.

### V4-Roadmap (NICHT in V3.1a implementieren)
- Iterativer Solver für Wellengleichgewicht und Choked-Flow-Matching
- θ_flow_lpt physikalisch in BraytonEngine eingebettet
- Drehzahlen aus Wellenbilanz statt als Input
- Map-Koeffizienten als trainable Parameter
```

## E.3 Akzeptanz-Definition Benchmark v1 (Tag 14)

- Lauffähiges CycleLayerV3 mit BraytonEngine (alle Stufen 0–6 grün)
- 4-Wege-Benchmark-Tabelle (v1, V2, CNN, V3) auf Test-Split
- θ-Trajectory-Plots: gelernte η_hpt, η_lpt vs. GT für 3 Test-Units
- LPT_flow_pred-Plot: gelernt vs. GT für 3 Test-Units
- 20%-Daten-Ablation als Tabelle: V3 vs. CNN bei reduzierter Trainingsmenge
- One-Pager: Architektur-Diagramm + zentrale Zahlen + Moat-Statement

## E.4 Pitch-Statement (für Outreach)

Das ist die Story, die V3.1a ehrlich erlaubt zu erzählen — formuliert für IMC 2026, PHME 2026, und Outreach an Sandfeld, Olga Fink, Baqué. Wichtig: zwei der drei GT-Korrelationen sind unsupervised (post-hoc), eine ist supervised. Das wird im Pitch sauber unterschieden.

> CycleLayer V3.1a ist eine differenzierbare Brayton-Cycle-Layer für 2-Spool-High-Bypass-Turbofans. Sie modelliert fünf Wirkungsgrad-Degradationen (Fan, LPC, HPC, HPT, LPT) als strukturierte θ-Parameter im Latent Space, mit Erhaltungssätzen als harte Constraints und vereinfachten parametrischen Maps statt proprietärer OEM-Maps.
>
> Die fünf θ-Wirkungsgrade werden **UNSUPERVISED** gelernt — getrieben nur durch Sensor-Konsistenz und RUL-Loss, ohne Zugriff auf Health-Parameter-Ground-Truth während Training. Im Post-hoc Vergleich gegen N-CMAPSS erreichen wir Pearson-Korrelationen > 0.7 für HPT_eff_mod und LPT_eff_mod — die θ-Trajektorien wurden während Training nie gegen diese GT-Werte optimiert.
>
> Zusätzlich liefert ein separater AuxHealthHead eine **supervised** Schätzung von LPT_flow_mod als Diagnose-Target. Diese ist ausdrücklich nicht Teil des unsupervised-Claims, sondern dient der Vollständigkeit der drei N-CMAPSS-Health-Parameter.
>
> Lokale Flow-Capacity-Degradation ist in V3.1a explizit aus der Physik-Layer ausgelagert, weil sie iteratives Komponenten-Matching erfordert. V4 wird Flow-Capacity mit differenzierbarem Solver in die Physics-Layer integrieren.
>
> Strategischer Differenzierungspunkt: Wir nutzen die Topologie und Erhaltungssätze, die jeder OEM kennt, ohne deren Component-Maps. Das macht den Ansatz für externe Service-Anbieter zugänglich, die keinen OEM-Datenzugang haben.

---

*— Ende der V3.1a Spezifikation —*

*Bei Unklarheiten: stop, frag Robert. Nicht ad hoc improvisieren.*
