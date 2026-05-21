---
name: upgrade-cuda
description: Upgrade the NVIDIA driver and CUDA toolkit on Ubuntu via the NVIDIA apt network repo. Use when the user asks to update / upgrade CUDA, install a newer toolkit, bump the driver, or follow the developer.nvidia.com/cuda-downloads instructions. Covers the reboot requirement, PATH-shadowing traps on AWS Deep Learning AMIs, and verification.
allowed-tools: [Bash, Read, Write, Edit, AskUserQuestion, WebFetch]
---

# Upgrade CUDA driver + toolkit

Use the NVIDIA apt **network** repo (deb_network) so future minor bumps are just `apt-get install`. Authoritative install commands per OS/arch come from https://developer.nvidia.com/cuda-downloads — fetch that page if unsure which `cuda-keyring` URL to use.

## Pre-flight

Check what's installed and what's running. Don't skip this — the running kernel module version is what matters, not the package version.

```bash
lsb_release -a                            # Ubuntu version (need 22.04 / 24.04 keyring URL)
nvidia-smi | head -5                      # current driver + max CUDA version it supports
cat /sys/module/nvidia/version            # running kernel module version
nvcc --version 2>/dev/null || true        # current toolkit nvcc in PATH
readlink -f /usr/local/cuda               # which toolkit /usr/local/cuda points at
nvidia-smi --query-compute-apps=pid,process_name --format=csv  # any live GPU processes?
```

If any process is using the GPU, stop here — the driver upgrade will need a reboot and you'll lose in-flight work.

## Step 1: Ensure the NVIDIA repo is registered

If `cuda-keyring` is already installed (`dpkg -l cuda-keyring`), skip to Step 2. Otherwise download the keyring deb from the NVIDIA download page (URL is OS-specific, e.g. `https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_<ver>_all.deb`):

```bash
wget <cuda-keyring-deb-url>
sudo dpkg -i cuda-keyring_*_all.deb
```

## Step 2: Refresh and inspect candidates

```bash
sudo apt-get update
apt-cache policy cuda-toolkit cuda-drivers       # see latest candidates
apt-cache search '^cuda-toolkit-1[0-9]-' | sort  # list installable toolkit metapackages
```

If you see warnings about duplicate `cuda-ubuntu*.list` sources (common when the repo was added manually then keyring added later), it's harmless — clean up after the upgrade if you care.

## Step 3: Install the toolkit and driver

Toolkits install side-by-side under `/usr/local/cuda-<major>.<minor>/`; the `cuda` and `cuda-<major>` metapackages flip `/usr/local/cuda` via `update-alternatives`.

```bash
sudo apt-get install -y cuda-toolkit-<major>-<minor> cuda-drivers
```

- `cuda-toolkit-<major>-<minor>` — the new toolkit. Does NOT remove the old one.
- `cuda-drivers` — branch-agnostic driver metapackage; upgrades to the newest driver that ships with the latest toolkit.
- **Blackwell GPUs (SM100, SM120)**: install `nvidia-open-<branch>` instead of (or in addition to) `cuda-drivers`. The proprietary modules don't support Blackwell. Confirm GPU with `nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader`.
- **Ada / Hopper / earlier**: proprietary modules from `cuda-drivers` are fine.

DKMS will build and sign the new kernel modules against the current kernel. Verify the build succeeded (look for "Building module(s)... done" in the apt output).

## Step 4: Reboot

The driver upgrade installs new kernel modules **on disk**, but the running modules can't be swapped in place — userspace libraries already moved to the new ABI, so `nvidia-smi` will fail with `Driver/library version mismatch` until reboot.

Unloading without reboot (`modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia`) usually fails because the persistence daemon and any cached CUDA contexts hold references. Just reboot.

```bash
sudo reboot
```

On AWS / shared infra, confirm with the user before rebooting — it drops SSH and may disrupt other workloads on the host.

## Step 5: Verify after reboot

```bash
nvidia-smi | head -5                  # new driver version + max CUDA
cat /sys/module/nvidia/version        # confirm running module matches the package
nvcc --version                        # confirm PATH points at new toolkit
readlink -f /usr/local/cuda           # should point at the new cuda-<major>.<minor>
```

All three should report the new version. If `nvcc --version` shows an old toolkit but `readlink -f /usr/local/cuda` is correct, see Step 6.

## Step 6: PATH-shadowing traps (important on AWS DLAMI)

AWS Deep Learning AMIs ship `/etc/profile.d/dlami.sh` which **hard-codes a specific `/usr/local/cuda-X.Y/bin` into `PATH` and `LD_LIBRARY_PATH`**, ahead of the alternatives-managed `/usr/local/cuda` symlink. After upgrading the toolkit, every new shell silently keeps using the old one.

Diagnose:

```bash
which -a nvcc
echo "$PATH" | tr ':' '\n' | grep cuda
grep -rE "cuda-[0-9]" /etc/profile.d/ ~/.bashrc ~/.profile ~/.zshrc 2>/dev/null
```

If `/etc/profile.d/dlami.sh` (or any other profile script) hard-codes a versioned path, patch it to use the `/usr/local/cuda` symlink so it follows `update-alternatives`:

```bash
sudo cp /etc/profile.d/dlami.sh /etc/profile.d/dlami.sh.bak-pre-cuda<new-ver>
# Edit: replace every /usr/local/cuda-<old-major>.<minor> with /usr/local/cuda
sudo tee /etc/profile.d/dlami.sh > /dev/null <<'EOF'
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/opt/aws-ofi-nccl/lib:/usr/local/cuda/lib:/usr/local/cuda/lib64:/usr/local/cuda:/usr/local/cuda/targets/x86_64-linux/lib/:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/lib:/usr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PATH=/usr/local/cuda/bin:/usr/local/cuda/include${PATH:+:$PATH}
EOF
```

Confirm in a fresh login shell:

```bash
bash -lc 'which nvcc && nvcc --version | tail -2'
```

The current shell's `$PATH` may still contain inherited old entries — they're benign as long as the symlinked entry comes first. New shells will be clean.

## Step 7: Cleanup (optional)

Older toolkits remain at `/usr/local/cuda-<old>/` and as `cuda-toolkit-<old-major>-<old-minor>` packages. Leave them unless disk pressure or audit requires removal — purging is irreversible without a reinstall, and some pinned environments (conda envs, container images) may still reference them. To remove:

```bash
sudo apt-get autoremove --purge cuda-toolkit-<old-major>-<old-minor> 'cuda-*-<old-major>-<old-minor>'
sudo rm -rf /usr/local/cuda-<old-major>.<old-minor>
```

## Compatibility notes for this repo

- vLLM, TRT-LLM, and FlashInfer are **forward-compatible across CUDA minor versions** (e.g. 13.1 → 13.2) — no rebuild needed. Major bumps (12.x → 13.x) almost always require new wheels.
- After a CUDA upgrade, re-run smoke benchmarks before trusting numbers. Driver-side autotune defaults (especially attention) can shift between driver branches.
- The CLAUDE.md guidance about Blackwell needing `-open` kernel modules applies here: if `nvidia-smi` post-reboot fails with "requires use of the NVIDIA open kernel modules", you installed the wrong driver variant — `apt-get install nvidia-open-<branch>` and reboot again.
