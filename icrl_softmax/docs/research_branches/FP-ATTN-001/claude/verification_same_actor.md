FP-ATTN-001 same-actor derived verification
==================================================================

1. Frozen inputs
  PASS  24 records (found 24)
  PASS  ATOL is the frozen 1e-4
  PASS  12 tasks per mixing
  PASS  both frozen mixing settings

2. Independently recomputed comparison statistics
  PASS  48 route-records compared (found 48)
  PASS  max |dQ| 1.076e-05 matches summary
  PASS  zero decision flips
  PASS  zero selected-eta flips
  PASS  numpy emissions 22 matches summary and the sealed 22/48
  PASS  literal emissions 22 equals the numpy count
  PASS  sealed regeneration validated for every route-record
  PASS  sealed E_Q reproduced exactly (max gap 0.0)
  PASS  sealed lower bounds reproduced exactly (max gap 0.0)
  PASS  every literal Qhat is finite
  PASS  no literal run tripped the divergence guard
  PASS  layer-0 diagnostics match exactly (max 0.0)

3. H5: the literal finite route is gate-free (executable)
  PASS  the finite route exposes no visited-query gate
  PASS  the masked exact route does carry a visited gate and null token
  PASS  every finite-route write weight is strictly positive (full support, no -inf mask)
  PASS  finite-route write attention is row-normalised
  PASS  the masked route's write attention does contain exact zeros (its mask)
  PASS  this evaluator introduces no -inf mask of its own

4. H6: no sealed file changed
  PASS  evaluate_fp_scale_002.py byte-identical to its hash at this run
  PASS  fixed_policy_expected_sarsa.py byte-identical to its hash at this run
  PASS  fixed_policy_expected_sarsa_scaled.py byte-identical to its hash at this run
  PASS  fixed_policy_variance_certificate.py byte-identical to its hash at this run
  PASS  model.py byte-identical to its hash at this run
  PASS  verify_variance_adaptive_certificate.py byte-identical to its hash at this run
  PASS  fixed_policy_expected_sarsa.py still matches its FP-SCALE-002 sealed hash (form: ['raw'])
  PASS  fixed_policy_variance_certificate.py still matches its FP-SCALE-002 sealed hash (form: ['lf'])

5. Replay of the sealed programs
  PASS  analyze_fp_attn_001.py replays with exit 0 (got 0)
  PASS  verify_variance_adaptive_certificate.py replays with exit 0 (got 0)
  PASS  verify_fp_scale_002_same_actor.py replays with exit 0 (got 0)

==================================================================
SUMMARY
  route-records compared        : 48
  max |literal Q - numpy Q|_inf : 1.076e-05  (ATOL 1e-04)
  numpy emissions               : 22 / 48
  literal emissions             : 22 / 48
  decision flips                : 0
  selected-eta flips            : 0
  sealed regeneration failures  : 0
==================================================================
RESULT: PASS
LIMITATION: same-actor derived verification only. No second actor
reconstructed this route; this is not reciprocal verification.
