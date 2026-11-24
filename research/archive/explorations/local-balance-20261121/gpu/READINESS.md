# Conditional GPU readiness

Read-only inspection on 2026-11-21 found an NVIDIA GeForce RTX 5070 Ti with 16,303 MiB total memory, 11,706 MiB free at inspection, and driver 610.88. The research interpreter is 64-bit Python 3.12.14. NumPy is available; PyTorch, SciPy, CuPy, ONNX Runtime and TensorFlow are absent.

A bounded search of known Conda/uv/Python locations and the Documents/ChatGPT workspaces found no reusable CUDA PyTorch environment or cached torch wheel. The only virtual-environment manifest found in these workspaces belongs to the current research environment. No private-directory-wide scan was performed. Pip reported no locally built torch wheel in its cache. Existing HTTP cache entries were not treated as an installed runtime.

The official [PyTorch CUDA 12.8 wheel index](https://download.pytorch.org/whl/cu128/torch/) lists a Python 3.12 Windows wheel, including torch 2.7.0+cu128. The official [PyTorch 2.7 release notes](https://pytorch.org/blog/pytorch-2-7/) document Blackwell support and CUDA 12.8 wheels. This established a plausible runtime route, before a locally verified installation. A Python HEAD request for wheel metadata failed with `URLError: [Errno 11001] getaddrinfo failed`; the failure is retained. No package was installed and no GPU model was executed during this initial inspection.

## Conditional real-network benchmark design

If the main scientific results justify proceeding and the root authorizes setup, use a separately isolated CUDA runtime and freeze an ordinary-trained ring classifier before timing. A moderate two-hidden-layer MLP (width 256 initially, ordinary Adam training) is a legitimate inference workload. A fixed width sweep may be prespecified before results, but unnecessary layers or repeated unused operations must not be added merely to manufacture a speedup. The 80-parameter pullback network and its dense parameter Gram matrix are not scaled up.

Compare raw, optimized full half-turn, global confidence with exactly 64/256 extra views, and a frozen meaningful calibrated selector. All timed outputs must correspond to real model logits on real transformed inputs. Verify outputs, selected counts and held-out classification/logistic risk. Report resident CUDA-event latency and synchronized host-to-device/deployment/device-to-host wall latency separately, with warmups, randomized order, repeated batches and precision explicitly fixed. Charge calibration and training separately. Include real routing and gather/scatter costs. A call-count reduction is not a latency speedup; no break-even exists when the limited policy is slower per batch.

At the initial inspection no runtime download size or setup duration was asserted because local metadata retrieval failed. Setup required a substantial CUDA wheel stack and additional disk space. That initial result was readiness only, not an efficiency result.

## Subsequent root decision

After the initial inspection, the root judged the old-model local-bound improvement sufficient to activate the conditional GPU branch. The root reported creation of an isolated `.venv-gpu` and an authorized official CUDA 12.8 torch 2.7.1 installation, including a 3.273 GB wheel download. This later setup does not erase the initial metadata-probe failure. Installation verification and any actual GPU benchmark are owned by the root and must be read from their separate records; this readiness note alone asserts no benchmark outcome.

The extension subsequently completed all nine real-model training, calibration, quality, equivalence and timing runs. The finished results are in `README.md` and `optimized/`; this document preserves the initial readiness investigation rather than replacing those completed results.
