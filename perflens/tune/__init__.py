"""PerfLens Tune: a compiler-in-the-loop optimizer for gpu-engine kernels.

Searches compile flags and launch parameters for one kernel. Every candidate
is compiled (its ptxas output parsed), checked against the CPU reference, and
timed on the GPU. Random, evolutionary, and agent-guided search are compared
at an equal evaluation budget, against exhaustive ground truth.

Kernel-side harness: gpu-engine/bench/ in kanish10/LLM-Inference-Server
(see its TUNING.md).
"""
