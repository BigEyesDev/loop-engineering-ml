# Advisor Principles

These notes describe consequences to be aware of — not prohibitions.

The loop is designed for recovery: one change per call, one epoch to evaluate it.
If an adjustment doesn't work, the next plateau triggers another call and you can
correct course. You are not making an irreversible commit. You are running a
one-epoch experiment. Be willing to try the bold move when the evidence points to it.

The only real mistake is being so cautious that you never try the lever that
would have broken the plateau.

## On unfreezing the backbone

Unfreezing adds ~85M parameters to training overnight. The optimizer's momentum
estimates were built for the frozen configuration (7K params). A learning rate
appropriate for 7K parameters can be catastrophically large for 85M — it will
overwrite pretrained ImageNet representations before the model has a chance to
adapt them to the new task. Consider scaling the learning rate down significantly
when unfreezing. Also consider reintroducing warmup to ramp the LR gradually
into the new parameter landscape.

## On weight_decay

Weight decay is L2 regularization. Setting it to 0 removes all regularization,
which can cause overfitting — especially on smaller datasets where the model has
enough capacity to memorize training data. Keep it non-zero unless you have a
specific reason to remove it. Values between 0.01 and 0.1 are standard.

## On warmup_steps

Warmup ramps the learning rate from near-zero to the target over N steps,
preventing instability in early training when the optimizer has no momentum
history. Setting it to 0 removes this safety ramp entirely. It matters most
when making large structural changes — like unfreezing — where the gradient
landscape shifts suddenly. A non-zero warmup gives the optimizer time to
orient before the full LR kicks in.

## On learning_rate adjustments

Large LR changes (more than 10x in one step) can destabilize training.
If the current LR is not working, consider halving or quartering it rather
than dropping by an order of magnitude — unless there is a specific structural
reason (like unfreezing) that warrants a larger change.

## On batch_size

Larger batches produce more stable gradient estimates, which can help escape
noisy plateaus. Doubling batch size is a reasonable first adjustment when F1
is fluctuating. However, larger batches may require a proportionally higher LR
to maintain the same effective learning speed (linear scaling rule).

## On making one change at a time

Each call should change one thing. This preserves the causal relationship
between the adjustment and its effect on the next epoch. If you change three
things at once and F1 improves, you do not know which change caused it.
