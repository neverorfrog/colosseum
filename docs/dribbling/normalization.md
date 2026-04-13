Why EMA normalization causes spikes
EMA = exponential moving average. The original code did:


new_mean = 0.99 * old_mean + 0.01 * batch_mean
new_var  = 0.99 * old_var  + 0.01 * batch_var
every training step. So _target_mean and _target_var are estimates of the distribution, built up slowly from the data the network has seen so far.

The problem: the target is used to compute the loss, but the target's normalization depends on running stats that themselves change every step. This is a feedback loop. Here's how it breaks:

Step 1 — initialization. At the start of Phase 2, _target_mean=0 and _target_var=1. So the "normalized target" is just the raw GT values. For a ball at x=1.5, the normalized target is 1.5.

Step 2 — EMA catches up. After many iterations the EMA converges toward the real distribution, say mean=0.5, var=0.25 for ball_x. Now target_norm = (1.5 - 0.5) / sqrt(0.25) = 2.0. The target for the same physical ball position changed from 1.5 to 2.0. The network was trained against 1.5, now it's evaluated against 2.0 → the loss jumps even though nothing physical changed.

Step 3 — burst resets. Imagine 2000 envs reset simultaneously (episode boundary). Freshly spawned balls are at a different distribution than mid-episode balls (e.g. all spawned in front, ball_x ≈ 1.2, ball_vx ≈ 0). For this one batch:

batch_mean jumps toward the spawn distribution.
batch_var collapses (less variety right after reset).
EMA momentum 0.01 × (epochs=5) ≈ 5% shift per iteration → visible jolt in stats.
When var shrinks, sqrt(var) shrinks, and dividing by it inflates normalized targets. Suddenly targets that were ~1 become ~3. The frozen-encoder predictions that used to match now have errors of ~2 in normalized units → MSE of ~4 → loss spike.

Step 4 — recovery. Over the next few iterations the EMA relaxes back toward the full-episode distribution, var grows, normalized targets shrink, loss drops.

Then it happens again on the next reset burst. That's exactly the pattern you saw: periodic spikes on adapt/loss, adapt/ball_pos, adapt/ball_vel, correlated with drops in valid_frac (reset bursts → fewer valid samples → stats shift harder → bigger spike). The encoder isn't actually failing — it's chasing a moving target created by its own normalizer.

There's a second subtler problem: EMA stats were updated inside num_learning_epochs of training, so each collection's data moved the stats 5× instead of 1×. That made the feedback loop even tighter.

Why fixed normalization fixes it
With fixed constants the equation is just:


target = (gt - [0.5, 0.0, 0.0, 0.0]) / [1.0, 0.5, 1.5, 1.5]
These numbers never change. Ever. The same physical ball state always maps to the same normalized target. The loss landscape is stationary — the network is chasing a fixed function of GT, not a function that moves whenever batch statistics drift.

The cost is that the constants may not perfectly match the true distribution — maybe the real std of ball_x is 1.2 instead of the assumed 1.0. That's fine. A 20% miscalibration just means one component contributes 20% more or less gradient than ideal. The network trains. What breaks training isn't imperfect normalization — it's non-stationary normalization.

Think of it like this: EMA normalization tries to solve "what are the statistics of my data?" while simultaneously training a network on that data. Fixed normalization says "I already know the approximate scale from physics, let me just use that." The first is clever but unstable; the second is crude but bulletproof.

Short version
Normalization = rescale each target component to unit scale so per-component MSE gradients are balanced.
EMA normalization = the rescaling constants drift every step, so identical physical states produce different loss values over time. Reset bursts shift the stats sharply, causing synchronized loss spikes and oscillation.
Fixed normalization = constants picked once from domain knowledge. Target function is stationary → clean, monotonic loss curves. Slight miscalibration is harmless.