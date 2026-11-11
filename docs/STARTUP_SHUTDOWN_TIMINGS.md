# Startup and Shutdown Stabilization

Measured on 2026-11-11 in `stab/paralelo`. Provider prerequisite:
`77d3a13` from main, cherry-picked as `371b65e` (cancellable Pocket TTS startup).

## Changes

- Controller startup no longer launches Pocket immediately. The window waits
  for its real QWebChannel bridge, then schedules prewarm 3000 ms later.
- Entering STANDBY or conversation still requests prewarm immediately and
  cancels the delayed timer. Duplicate concurrent workers remain blocked.
- Shutdown cancels the delayed timer and stops the TTS manager when a Pocket
  prewarm is in progress, releasing the worker without waiting for model load.
- `app.main()` checks voice dependencies but no longer probes audio devices.
  Microphone tests and device selection remain on explicit test/capture paths.

## Before / After

Fresh-process measurements with the real app, Qt/QWebEngine/QWebChannel and
Pocket provider, using `QT_QPA_PLATFORM=offscreen`, isolated temporary memory
and workspace, the same application Python and existing local model caches.
No screenshots, paid model calls, package installation or settings-file edits.

The timing harness was outside the repository. It measured from script entry
to a ready JS bridge, then requested a normal window close. Shutdown is measured
from that request until `app.main()` returns. The extra wait for prewarm in the
second scenario is not counted as shutdown latency.

| Scenario / metric | Before (ms) | After (ms) |
| --- | ---: | ---: |
| Early close: UI bridge ready | 4608 | 2713 |
| Early close: startup microphone probe | 2189 | not called |
| Early close: shutdown | 4491 | 10 |
| Close during prewarm: UI bridge ready | 4852 | 2485 |
| Close during prewarm: startup microphone probe | 2225 | not called |
| Close during prewarm: shutdown | 4190 | 58 |

Before, prewarm started before bridge readiness and was still active at close.
After, early close occurred without starting Pocket; in the second run prewarm
started 3005 ms after readiness, was active at close and was cancelled.
An earlier pair of post-change runs produced UI readiness at 2604--2831 ms,
early shutdown at 28 ms and shutdown during prewarm at 53 ms. The table uses
the final-code repetition, run after pytest had finished to avoid CPU contention.

These are individual runs per scenario, not statistical medians or cold-boot
guarantees. Offscreen Chromium produced graphics mailbox warnings; these
measurements validate lifecycle timing, not visual rendering performance.
Session-close persistence had no conversation to summarize in these runs.

## Regression Coverage

- `tests/test_tts_prewarm.py`: readiness and delay, cancelled startup timer,
  immediate voice entry, no duplicate worker, shutdown during cancellable load.
- `tests/test_startup_lifecycle.py`: no startup microphone probe and notification
  only after bridge readiness, with late callbacks ignored during close.
- `tests/test_pocket_tts_cancellable_start.py`: actual provider lifecycle using
  subprocess doubles, including early EOF and startup cancellation.

The existing manual microphone-test and recording paths are not removed.

Validation: **1992 passed, 1 xfailed** in the full suite; **25 passed** in the
focused lifecycle/QWebChannel checks. Compilation and diff checks passed.
Only the prerequisite cherry-pick is committed; the app/controller/window
changes and their tests/documentation remain uncommitted for review.
