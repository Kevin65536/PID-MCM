# B4 config queue

No formal configuration is frozen. Development configs must not open protected
indices or write outside this method's `runs/` root.

The executable public-development v1 contract is shared at
[`../../single_modal_eeg/configs/public_performance_v1.yaml`](../../single_modal_eeg/configs/public_performance_v1.yaml).
It freezes PREST-16, task windows, native-electrode panels, public split
fingerprints, probe search space, seeds, and REFED as unsupported. It is not a
formal protected-evaluation unlock.
