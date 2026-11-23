# Forecasting tab follow-up

## Issue to revisit

The three-shift synthetic demo can trigger an ADWIN update, but after the update
the adaptive 24-hour-ahead forecasts may deviate more from actual demand than
the frozen static forecasts (screenshot at 2025-01-11 00:00 UTC). This is a
known demo limitation to investigate before presenting forecast quality.

## Current explanation / caveat

For the specially named `demo_drift_3events` file, the dashboard currently uses
`increase_only` adaptation so a meaningful ADWIN error increase visibly
re-trains the adaptive model. That demo path intentionally does not compare the
candidate against the deployed model before accepting it. ADWIN identifies a
change in forecast-error distribution; it does not guarantee that retraining
will improve accuracy. Retraining on a short/abrupt synthetic regime mixed with
the trailing 60 days can move the MLP weights and re-fitted scalers in a way
that hurts subsequent forecasts. This is a likely explanation, not yet a
root-cause finding.

## Follow-up investigation

- Inspect each accepted update's training window, candidate/current validation
  errors, scaler fit ranges, and subsequent static-vs-adaptive MAE/RMSE.
- Compare the error-increase-only demo mode with validated adaptation and
  fine-tuning; retain the frozen baseline and identical observed demand.
- Keep the demo policy clearly labeled and separate from normal validated
  replay. Do not present visible model change as evidence of improved accuracy.
- Do not tune against the final evaluation period; report whether adaptation
  helps or hurts after each event.
