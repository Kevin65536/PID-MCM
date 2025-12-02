This is a simulation to analyze PID in noisy background.

Lets do some Thought experiment. For partial information decomposition, what result will we get analysing two variable that are exactly the same before added with different ramdom noise? For eeg signal with high noise-information ratio, what result could we expect decomposing multi-channal siganl? 

1. Identical Variables with Independent Noise

If you have two variables $X_1$ and $X_2$ that are identical copies of a source $S$ but with different random noise:

- Redundancy: This will be the dominant component. Since both variables "mean" the same thing (the source $S$), the information they share is simply the information contained in the noisier of the two (or both if equal).
- Synergy/Unique: There is a small but important Synergistic (or Unique, depending on definition) gain. Combining $X_1$ and $X_2$ allows you to average out the independent noise, recovering more information about $S$ than either variable alone.
Simulation Result: With high noise, we saw $I(S; {X_1, X_2}) > I(S; X_1)$, confirming this gain.

2. High-Noise EEG Signals (Multi-channel)

For EEG signals where the noise-to-information ratio is high:

- Single Channel: Contains almost zero information ($I(S; X_i) \approx 0$).
- Multi-Channel: Can contain significant information ($I(S; X_{joint}) \gg 0$).
- PID Result: This manifests as Pure Synergy.
    - The information is not in any single channel (Redundancy $\approx 0$).
    - The information only emerges when considering the pattern across many channels.
    - Note: Standard MI estimators often fail here due to the "curse of dimensionality," so specialized decoding-based PID methods are recommended.