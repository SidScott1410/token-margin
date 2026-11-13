# Pre-registration: asking price vs measured cost floor for open-weight inference

Written before the analysis ran. RESULTS.md reports what happened against this.

## Question

For a model sold by several providers, how much of each seller's asking price is explained
by the measured cost of serving it on rented silicon, and how much is margin or mispricing?

## Inputs and their provenance

| input | source | provenance |
|---|---|---|
| Seller asking prices | tokenprice `prices/latest.json`, per-provider endpoints | api, read_at in file |
| Decode and prefill timing per silicon, batch, context | reckon measured corpus, schema 5, July 2026 | measured |
| GPU hourly rate | `prices.yaml` | manual (paid) where rented; page (list) otherwise |
| Token mix, utilization | this file | convention |

## Cost floor

For a corpus cell (silicon, model, batch B, prompt P, output O) with measured TTFT and TPOT:

    time_per_batch_ms = TTFT + TPOT * (O - 1)
    cost_per_request  = rate_usd_per_hr / 3600 * time_per_batch_ms / 1000 / B / u
    floor_usd_per_M   = cost_per_request / (P + O) * 1e6

u is utilization, the share of wall time the GPU is serving a full batch. Base case u = 0.6;
sensitivity at 0.3 and 0.9. Headline cell: B = 32, P ~ 2048, O = 256. All cells reported.

The seller's price at the same mix:

    price_per_request = P * input_per_M / 1e6 + O * output_per_M / 1e6
    seller_usd_per_M  = price_per_request / (P + O) * 1e6
    implied_margin    = 1 - floor_usd_per_M / seller_usd_per_M

The floor excludes everything but the GPU hour: no CPU, network, storage, staff, or
failed-request cost. It is a floor, not a cost estimate. Margins reported against it are
upper bounds.

## Model mapping

| corpus model | sellers' model id | assumption |
|---|---|---|
| llama3-8b | llama-3.1-8b-instruct | Same architecture and parameter count; decode floor transfers. Base Llama 3 8B is no longer sold on OpenRouter. |
| qwen3-30b-a3b | qwen3-30b-a3b and qwen3-30b-a3b-instruct-2507 | Same architecture; the 2507 variant is a post-training update. |

## Hypotheses

H1. No seller's price is below the cost floor on the cheapest silicon at B = 32, u = 0.9.
    Falsified if any seller is under the floor: a loss leader or a mispricing exists.

H2. Dispersion of seller prices (max/min) exceeds dispersion of cost floors across silicon
    (max/min over silicon at the headline cell). Falsified if hardware choice alone could
    explain the price spread.

H3. Implied margin at the median seller exceeds 50% at u = 0.6 for both models.
    Falsified if open-weight serving at list GPU rates is a thin-margin business.

H4. Seller price is not explained by declared quantization (fp8 vs bf16 sellers do not
    differ in price by more than the within-group spread). Underpowered at n = 4; reported
    as descriptive only until Phase 2.

## Known limits, stated in advance

- n = 4 sellers per model in Phase 1. Case study, not a distribution. Phase 2 measures
  gpt-oss-120b (26 sellers) and qwen3.6-35b-a3b (10) on the same harness.
- Sellers do not run on RunPod on-demand rates. Reserved and owned capacity is cheaper;
  the floor is therefore conservative in the seller's favor at u fixed, and the true
  utilization is unknown. Both cut the same way: reported margins are upper bounds on
  what a RunPod-renting seller could earn, not estimates of what incumbents earn.
- Groq sells llama-3.1-8b on its own silicon. No corpus cell applies; it is reported
  against the cheapest GPU floor and flagged.
- July timing data, July paid rates, November list rates and November asking prices.
  B200's list moved 27% between July and November; drift is real and every number is dated.
