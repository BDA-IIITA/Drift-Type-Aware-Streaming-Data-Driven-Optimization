# Configurable Phase 1 scenario assumptions

`configs/base.yaml` defaults the metadata to DE-LU, 2021–2024 because no
existing GridDRIFT demand file is present in this checkout. It does not download
or manufacture ENTSO-E demand.

Solar and wind profiles are simulated. Hydro begins with a finite 336 MWh budget
that later rolling-horizon simulation must carry forward; it must not reset each
day or solve. Conventional generation is capacity/ramp limited. Emergency grid
power is more expensive, and any remaining deficit will later receive a VOLL
penalty of 10,000 EUR/MWh.

The BESS configuration reserves a 3–27 MWh usable band within a 30 MWh battery.
The terminal-SOC policy is intentionally deferred to the dispatch phase, where
its feasibility can be checked against the solved horizon instead of imposed as
an unconditional configuration rule.
