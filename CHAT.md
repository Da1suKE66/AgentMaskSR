# Chat

This file preserves the key research direction for AgentMaskSR.

## Summary

The project goal is to convert a masked generative image model from:

```text
generate one image
```

into:

```text
selectively complete unknown detail tokens under a known low-resolution observation
```

The low-resolution input provides the observable structure. Unknown high-frequency details and expanded regions are treated as masked content to be refined by a frozen visual prior.

## Final System Positioning

Recommended architecture:

```text
User instruction + LR image
        |
        v
Agent / planner
        |
        v
Structured control plan
        |
        v
Training-free controller
        |
        v
Meissonic masked image refinement
        |
        v
LR consistency / boundary consistency / tile decoding
```

Responsibilities:

- Agent: understand intent and produce mode, alpha, mask policy, consistency weights, and prompt.
- Meissonic: provide the frozen high-resolution masked generative visual prior.
- Controller: enforce observation consistency, adaptive masking, outpaint boundaries, tiling, and early commit.

## Current Implementation

The first AgentMaskSR implementation uses Meissonic's existing inpaint interface:

```text
init_observation.png + mask_refine.png + prompt -> refined image
```

This is a practical bridge to the full token-level method. It lets the controller be tested before modifying scheduler internals.

## Research Contributions to Preserve

1. Agent-guided controller.
2. Observation-constrained masked refinement.
3. Adaptive spatial-frequency unmasking.
4. Detail-outpaint controller.
5. Tile-global sparse decoding.
6. Context reuse / early commit.

## Important Constraint

Do not let a closed API generate the final image. A VLM/API can be used only for planning and region understanding. The final image should come from Meissonic plus the training-free controller so the paper contribution remains the masked token refinement method.
