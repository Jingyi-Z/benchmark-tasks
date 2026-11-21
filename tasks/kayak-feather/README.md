# Feather a 4-piece kayak paddle to 30 degrees right-hand

A 210 cm Aqua Bound Sting Ray Hybrid 4-piece Posi-Lok paddle lies at 0 degrees feather in two canoe blocks. The robot must
press both centre-ferrule buttons together, pull the halves apart past the angle cogs, turn the right half 30 degrees
right-hand, push it home until both catches snap behind the clip lip, and let go. One button alone does not release the
joint.

## Run a trial

Inside the Isaac Lab-Arena environment with `yam-isaac-lab-arena` installed (all `.py` files in this folder next to
`task_demo.py`):

    KF_MODE=roll KF_LAYOUT=channel KF_CO=0.0001 KF_BOUND=lo \
    inspect-robots run --embodiment yam_arena --policy dream -E task=task_demo.py \
      -E object_jitter=0 -E control_hz=20 -E cam_width=640 -E cam_height=352 \
      --instruction "Set the paddle to 30 degrees right-hand feather." \
      --scorer task_score --max-steps 1000

For the scripted demonstration, use `--policy yam_scripted` with the same command.

- Run limit: 1,000 control steps (50 s at 20 Hz). The scripted demo reaches 1.0 at step 550.
- Runtime per trial: about 3.5 min (demo) and 7.5 min (Dream-1, 1,000 steps) on one RTX 5090.
- No seeds. Object jitter is 0, so the start pose is fixed.
- `KF_CO=0.0001` sets a 0.1 mm solver contact offset. The default 0.5 mm is wider than the 0.3 mm bore clearance, and
  speculative contacts then hold the shaft in the bore (shown natively).

## Score (end state, every frame of the final 1 s window)

- **1.0:**
  - the right blade is turned 30 +- 5 degrees right-hand relative to the left blade;
  - the ferrule is seated, coaxial within 0.5 mm, with both shafts straight within 2 degrees;
  - each catch hook overlaps the clip lip by 0.2 mm or more, and sits in its groove;
  - both shafts rest in their block channels;
  - both hands are 80 mm or more from the paddle, blades included;
  - no blade-tip, blade-edge or ferrule mark moves 5 mm or more.
- **0.5:** the same within 5 degrees of another stock setting: 15, 45 or 60 right, or 15 to 60 left.
- **0:** everything else, including unchanged at 0 degrees, unseated, off a setting, or still held.
- The live scorer is `task_demo._score_step`.
- Real trial checks:
  - feather: protractor on the two blade faces;
  - seat: use a feeler gauge to verify a collar gap of 0.5 mm or less, show both buttons fully returned, and apply a 20 N axial pull with a spring scale to confirm the halves stay locked;
  - stillness: video.

## Model

- **Sourced:**
  - shaft OD 30.0 mm (Aqua Bound FAQ, regular carbon 1.18 in);
  - the maker lists 850 g and 16.5 x 46 cm blades for this 210 cm paddle. The sim uses 865 g and rigid 16 x 46 x 0.6 cm rectangular blade proxies;
  - index set 0 / 15 / 30 / 45 / 60 degrees, left and right, and the two-button release, where one button alone does not disengage (Aqua Bound Posi-Lok page);
  - dual push tabs on resilient arms with catch hooks on a full-circle clip lip, exterior mating cogs, and the press-pull-rotate-slide sequence (US 8,801,483 B2).
  - Canoe blocks: NRS Universal Canoe Blocks, 4 x 4 x 7 in, 38.1 mm channel.
- **Inferred and bounded:** these are tested natively at both bounds (`ferrule_proto.BOUNDS`):
  - ferrule OD 36 to 39 mm;
  - key depth 6 to 10 mm;
  - tab diameter 8 to 12 mm;
  - tab protrusion 2.5 to 4 mm;
  - hook overlap 1.0 to 1.5 mm;
  - key width 1.6 to 2.0 mm.
- **Unmeasured:** tab spring force is assumed (400 N/m). Rigid rectangular blade proxies preserve span and feather axis; channel profile simplified.
- Geometry checks: `ferrule_proto.py` (gates), `audit_overlap.py` and `audit_sweep.py` (every setting, stroke and button state).

## Reset

Press both buttons, turn the paddle back to 0 degrees, push home until both buttons click out, and lay it in the two
blocks: tab half's block 450 mm and ferrule half's block 500 mm from the ferrule, ferrule mouth 350 mm in front of the
robot bases. Nothing is consumed.

## Video and screenshot view

The submitted videos and screenshot come from an output-only review camera, `review_cam`. It sits at the top camera's
calibrated pose with a wider lens (0.63 x focal length), so both full blades and the centre ferrule are in frame.

- It is added by `review_task.py`, which imports the scored `task_demo.py` unchanged.
- It is never part of the policy's observation (top and wrist cameras only).
- The wide scripted demo matches all 561 stored actions and all logged telemetry of the original 1.0 demo.
- The wide Dream-1 is a separate 1,000-step run on the same frozen task and policy inputs. Its logged paddle state and scorer fields match the original at 111/111 samples; the actions and jaw readings vary. The wide Dream-1 recording itself scored 0.0.
- To record it, run the same command with `-E task=review_task.py` and `KF_REVIEW_DIR=<folder>`; one JPEG is written per control step.
