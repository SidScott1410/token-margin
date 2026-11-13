# Results, Phase 1 (existing corpus, n = 4 sellers per model)

Pre-registered in HYPOTHESIS.md. Inputs: reckon measured corpus (July 2026, schema 5),
tokenprice feed read 2026-11-13T00:24:58Z, GPU rates in prices.yaml (paid July rates for
L40S, H100 PCIe, B200; RunPod list 2026-11-12 otherwise). Headline cell: batch 32,
prompt ~2048, output 256, utilization 0.6. Full grid in results/floors.csv; sensitivity
in results/summary_u30.csv and summary_u90.csv.

## Statement 1. Two of four Llama 3.1 8B sellers price below the rented-GPU cost floor. H1 falsified.

| seller | quant | $/M at mix | margin vs cheapest floor (MI300X, $0.070) | what it would take |
|---|---|---|---|---|
| deepinfra | fp8 | 0.022 | -217% | never breaks even at RunPod MI300X rate at any batch: price is below the per-sequence asymptote of the fitted batch curve. Breaks even at B=32 only if the GPU costs $0.74/hr. |
| groq | LPU | 0.053 | -32% | batch ~312 on MI300X (extrapolated 10x past the measured range), or $1.78/hr at B=32. Groq runs its own silicon; no corpus cell applies. |
| cloudflare | fp8 | 0.167 | 58% | |
| coreweave | bf16 | 0.220 | 68% | |

At u = 0.9 the floor drops to $0.047 and deepinfra is still 2x under it. The finding is
robust to utilization. It is not robust to hardware cost: a seller amortizing an owned
H100 or MI300X at roughly $0.75 to $1.00 per GPU-hour (plausible for owned capacity at
4-year depreciation and high utilization) clears deepinfra's price. The paper's claim is
therefore: **deepinfra's Llama 3.1 8B price is consistent only with owned hardware at
amortized cost near $0.75/hr, not with rented capacity at any batch size.**

## Statement 2. Price dispersion exceeds floor dispersion. H2 confirmed, both models.

Llama 3.1 8B: sellers span 9.9x; cost floors across five silicon types span 3.2x
(MI300X $0.070 to H100 SXM $0.226). Hardware choice explains at most a third of the
price spread in log terms. Qwen3-30B-A3B: sellers 1.43x, floors 1.01x (one silicon
measured; the spread is between A100 cells, not across silicon). Underpowered.

## Statement 3. Median implied margin is below 50% at u = 0.6. H3 falsified at base case.

Llama 3.1 8B median seller margin vs cheapest floor: 36% (u 0.6), 57% (u 0.9), -28%
(u 0.3). Qwen3-30B-A3B: 20% / 46% / -61%. Against the dearest measured silicon (H100 SXM
at list), every Llama seller is at or below the floor. On rented on-demand GPUs at these
batch sizes, open-weight serving is a thin-margin business; margins exist only on cheap
silicon run hot, or on owned hardware.

## Statement 4. Quantization does not order price. H4 descriptive only.

Llama: the bf16 seller (coreweave) is the dearest; the fp8 sellers span 0.022 to 0.167.
Qwen: all fp8 or undeclared, 0.113 to 0.162. n = 4; no test run.

## Confounds, in order of size

1. **Batch ceiling.** The corpus stops at B = 32. For an 8B model, production batches
   are larger; the floor at B = 32 is an upper bound on the true floor. The break-even
   fit addresses this for below-floor sellers; it is an extrapolation past the data.
2. **Rented vs owned.** On-demand RunPod rates are the most expensive way to hold a GPU.
   Every margin here is an upper bound on a renting seller's margin and says nothing
   directly about an owner's. The break-even GPU rate column is the owner-side number.
3. **Architecture equivalence.** Corpus timings are for Llama 3 8B; sellers list
   Llama 3.1 8B. Same parameter count and layer shapes; decode floor transfers. Not
   re-measured.
4. **Dates.** July timings and paid rates, November list rates and asking prices. B200
   list moved 27% in that window. Every number carries its date.
5. **Floor scope.** GPU-hour only. No CPU, network, storage, staff, egress, or failed
   requests. Real margins are lower than every figure here.
6. **n = 4.** A case study. Phase 2 (gpt-oss-120b, 26 sellers; qwen3.6-35b-a3b, 10)
   on the same harness is what makes any of this a distribution.

## Reproduce

    python margin.py            # u=0.6, batch 32
    python margin.py --u 0.9

No network, no fitted parameters beyond the per-silicon linear batch fit used only for
break-even extrapolation.
