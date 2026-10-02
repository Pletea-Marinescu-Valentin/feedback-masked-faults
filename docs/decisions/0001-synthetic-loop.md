# 0001. Synthetic closed loop for verifying C1

- Status: accepted
- Date: 2026-10-03

## Context

C1 makes quantitative claims (masking, the command shift per fault, delay as a
function of information and long-run variance) that need a loop with known
fault onset, magnitude and noise. Public simulated datasets have noise-free
sensors and idealized control, so they cannot calibrate sigma^2 or rho.

## Decision

`fmf.sim` implements one SISO loop, defaulting to an AHU cooling coil on
supply-air temperature:

- Plant: first-order lag (tau = 120 s), discretized exactly under zero-order
  hold at the sampling period dt = 60 s. Steady state
  y = y_free + sign * theta * phi(u) / (mdot * cp), with y_free the coil inlet
  temperature plus an unmeasured AR(1) disturbance.
- Controller: velocity-form PI with output clamping to [u_min, u_max];
  lambda tuning at the nominal point (kp = 0.0335 1/K, ti = 120 s, closed-loop
  poles 0.76 and 0.45 per sample).
- Valve: normalized characteristic phi (linear or equal-percentage, R = 50).
- Faults: additive CV sensor bias b(t) and capacity loss
  theta = theta0 * (1 - delta(t)), both as arbitrary trajectories.
- Noise: white CV sensor noise, quantization, AR(1) unmeasured disturbance.
- Baseline: oracle u_hat(z) = phi^{-1}(L(z) / theta0) from the measured
  context.

## Consequences

- Steady states match `fmf.theory.masking` exactly (tests to 1e-6), so
  deviations in later experiments come from dynamics or noise, not from the
  model.
- The control period equals the logging period; the plant sees a one-sample
  transport delay only. Loops logged slower than they run need a separate
  inner step.
- Small-signal results (e.g. long-run variance of u equal to lrv / P(0)^2,
  independent of PI tuning) hold only for well-damped loops. A strongly
  curved valve combined with a poorly damped loop leaks high-frequency power
  into low frequencies through the nonlinearity (observed: 9x the linear
  prediction at twice the nominal gain with the equal-percentage valve).
- No economizer, mixing box or fan dynamics; fan-off periods are not modeled.
