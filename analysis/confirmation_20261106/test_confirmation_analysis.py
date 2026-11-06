"""Independent synthetic arithmetic controls; never reads scientific outcomes.

Expected contrasts are copied from the locked protocol, not inferred from the
implementation under test.  Expected probabilities/areas below are analytical.
"""
from __future__ import annotations

import copy
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import confirmation_analysis as ca

ROOT = Path(__file__).resolve().parent
EXPECTED_IDS = (
    "Warehouse Direct minus Fixed",
    "Warehouse Direct minus Lagrangian",
    "Warehouse Direct minus shield-aware",
    "RWARE Direct minus Fixed",
    "RWARE Direct minus Lagrangian",
    "RWARE Direct minus shield-aware",
    "Warehouse execution Direct minus Fixed below capacity",
    "Warehouse execution Direct minus Lagrangian below capacity",
    "Warehouse execution Direct minus shield-aware below capacity",
    "Warehouse execution Direct minus Fixed overload",
    "Warehouse execution Direct minus Lagrangian overload",
    "Warehouse execution Direct minus shield-aware overload",
)


def rational(value):
    """Read exact output without using the displayed decimal as an oracle."""
    return F(value["numerator"], value["denominator"])


def p_record(contrast_id, p):
    p = F(p)
    return {
        "contrast_id": contrast_id,
        "p_value": {
            "numerator": p.numerator,
            "denominator": p.denominator,
            "decimal": float(p),
        },
    }


def complete_records(probabilities=None):
    values = probabilities if probabilities is not None else [F(1)] * 12
    return [p_record(name, p) for name, p in zip(EXPECTED_IDS, values)]


class HypervolumeArithmetic(unittest.TestCase):
    def test_single_rectangle_and_two_step_union(self):
        self.assertEqual(ca.hypervolume_exact([(F(1, 100), F(3, 10))], F(1, 20)), F(6, 25))
        # Independent area: (.02-.01)*.3 + (.05-.02)*.6, divided by .05.
        points = [(F(1, 100), F(3, 10)), (F(1, 50), F(3, 5))]
        self.assertEqual(ca.hypervolume_exact(points, F(1, 20)), F(21, 50))

    def test_dominated_equal_cost_and_outside_points(self):
        base = [(F(1, 100), F(3, 10))]
        augmented = base + [(F(1, 50), F(1, 5)), (F(1, 100), F(1, 4)), (F(3, 50), F(1))]
        self.assertEqual(ca.hypervolume_exact(augmented, F(1, 20)), F(6, 25))
        self.assertEqual(ca.hypervolume_exact(list(reversed(augmented)), F(1, 20)), F(6, 25))

    def test_empty_full_zero_utility_and_cost_boundary(self):
        self.assertEqual(ca.hypervolume_exact([], F(1, 20)), F(0))
        self.assertEqual(ca.hypervolume_exact([(F(0), F(1))], F(1, 20)), F(1))
        self.assertEqual(ca.hypervolume_exact([(F(0), F(0)), (F(1, 20), F(1))], F(1, 20)), F(0))

    def test_integer_count_rectangle_identity(self):
        # D=1000 decisions; cost limit200/D; cost100/D; utility400/D.
        # Count arithmetic gives ((200-100)*400)/(200*1000) =1/5.
        self.assertEqual(ca.hypervolume_exact([(F(100, 1000), F(400, 1000))], F(200, 1000)), F(1, 5))

    def test_exact_ties_survive_split_integration(self):
        # Floating integration previously differed at about2e-19 for this pair.
        dominant = (F(1, 40960), F(53, 40960))
        dominated = (F(3, 40960), F(29, 40960))
        self.assertEqual(
            ca.hypervolume_exact([dominant, dominated], F(1, 20)),
            ca.hypervolume_exact([dominant], F(1, 20)),
        )

    def test_mean_points_and_mean_seed_hypervolume_do_not_commute(self):
        a = [(F(0), F(1, 5)), (F(1, 10), F(1)), (F(0), F(2, 5))]
        b = [(F(1, 10), F(1)), (F(0), F(1, 5)), (F(1, 10), F(9, 10))]
        seed_areas = [ca.hypervolume_exact([x, y], F(1, 20)) for x, y in zip(a, b)]
        self.assertEqual(seed_areas, [F(1, 5), F(1, 5), F(2, 5)])
        means = [(sum(p[0] for p in arm) / 3, sum(p[1] for p in arm) / 3) for arm in (a, b)]
        self.assertEqual(ca.hypervolume_exact(means, F(1, 20)), F(8, 45))
        self.assertEqual(sum(seed_areas) / 3, F(4, 15))
        self.assertNotEqual(ca.hypervolume_exact(means, F(1, 20)), sum(seed_areas) / 3)

    def test_invalid_coordinates_and_reference_rectangles_refused(self):
        for points, bound in [
            ([(F(-1, 100), F(1, 2))], F(1, 20)),
            ([(F(0), F(101, 100))], F(1, 20)),
            ([(F(101, 100), F(1, 2))], F(1, 20)),
            ([(F(0), F(-1, 100))], F(1, 20)),
            ([], F(0)),
            ([], F(2)),
        ]:
            with self.subTest(points=points, bound=bound):
                with self.assertRaises((ValueError, TypeError)):
                    ca.hypervolume_exact(points, bound)


class WholeClusterSignFlip(unittest.TestCase):
    def assert_probability(self, differences, p, exceedances):
        result = ca.exact_sign_flip(differences)
        self.assertEqual(result["assignments"], 2**20)
        self.assertEqual(result["exceedances"], exceedances)
        self.assertEqual(rational(result["p_value"]), p)
        self.assertEqual(rational(result["statistic_abs_mean"]), abs(sum(differences) / 20))

    def test_all_zero_clusters_include_all_assignments(self):
        self.assert_probability([F(0)] * 20, F(1), 2**20)

    def test_twenty_equal_nonzero_differences_have_two_extreme_signs(self):
        self.assert_probability([F(1, 3)] * 20, F(2, 2**20), 2)

    def test_inclusive_exact_ties_and_zero_cluster_multiplicity(self):
        # Four nonzero equal magnitudes:2 sums of magnitude4 plus8 of magnitude2.
        # Sixteen zero coordinates replicate each of those10 assignments2^16 times.
        self.assert_probability([F(1), F(1), F(1), F(-1)] + [F(0)] * 16, F(5, 8), 10 * 2**16)

    def test_microscopic_rational_differences_preserve_exact_tail(self):
        unit = F(1, 10**30)
        self.assert_probability([unit, unit, unit, -unit] + [F(0)] * 16, F(5, 8), 10 * 2**16)

    def test_zero_observed_sum_has_probability_one(self):
        self.assert_probability([F(1), F(-1)] * 10, F(1), 2**20)

    def test_scientific_cluster_count_must_be_exactly_twenty(self):
        for n in (0, 19, 21):
            with self.subTest(n=n):
                with self.assertRaises((ValueError, TypeError)):
                    ca.exact_sign_flip([F(1)] * n)

    def test_inexact_or_nonfinite_difference_inputs_refused(self):
        for invalid in (0.1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaises((ValueError, TypeError)):
                    ca.exact_sign_flip([F(1)] * 19 + [invalid])


class GlobalHolmFamily(unittest.TestCase):
    def test_exact_protocol_membership(self):
        self.assertEqual(tuple(ca.EXPECTED_CONTRASTS), EXPECTED_IDS)

    def test_manual_stepdown_staircase_and_monotonicity(self):
        raw = [F(1, 1000), F(1, 100), F(1, 50), F(3, 100), F(1, 25), F(1, 20),
               F(3, 50), F(7, 100), F(2, 25), F(9, 100), F(1, 10), F(1, 5)]
        expected = [F(3, 250), F(11, 100), F(1, 5), F(27, 100), F(8, 25), F(7, 20)] + [F(9, 25)] * 6
        result = ca.holm12(list(reversed(complete_records(raw))), require_complete=True)
        self.assertEqual(result["status"], "COMPLETE_GLOBAL_HOLM12")
        by_id = {row["contrast_id"]: row for row in result["contrasts"]}
        for position, (name, adjusted) in enumerate(zip(EXPECTED_IDS, expected)):
            self.assertEqual(rational(by_id[name]["adjusted_p_value"]), adjusted)
            self.assertIs(by_id[name]["reject"], position == 0)

    def test_equal_p_values_receive_equal_adjustments(self):
        result = ca.holm12(complete_records([F(1, 100)] * 12), require_complete=True)
        for row in result["contrasts"]:
            self.assertEqual(rational(row["adjusted_p_value"]), F(3, 25))
            self.assertIs(row["reject"], False)

    def test_complete_family_uses_inclusive_alpha_point_zero_five(self):
        result = ca.holm12(complete_records([F(1, 240)] + [F(1)] * 11), require_complete=True)
        row = next(r for r in result["contrasts"] if r["contrast_id"] == EXPECTED_IDS[0])
        self.assertEqual(rational(row["adjusted_p_value"]), F(1, 20))
        self.assertIs(row["reject"], True)

    def test_missing_family_is_refused_for_final_adjustment(self):
        for size in (0, 3, 6, 11):
            with self.subTest(size=size):
                with self.assertRaises((ValueError, TypeError)):
                    ca.holm12(complete_records()[:size], require_complete=True)

    def test_partial_family_is_descriptive_without_adjusted_or_reject_fields(self):
        records = complete_records([F(1, 1000)] * 12)[:3]
        result = ca.holm12(records, require_complete=False)
        self.assertEqual(result["status"], "DESCRIPTIVE_PARTIAL_FAMILY")
        self.assertEqual(set(result["pending_contrasts"]), set(EXPECTED_IDS[3:]))
        self.assertEqual(result["contrasts"], records)

        def check(value):
            if isinstance(value, dict):
                self.assertNotIn("adjusted_p_value", value)
                self.assertNotIn("reject", value)
                for child in value.values():
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)
        check(result)

    def test_duplicate_and_unknown_contrasts_always_refused(self):
        duplicate = complete_records()
        duplicate[-1] = copy.deepcopy(duplicate[0])
        unknown = complete_records()
        unknown[-1]["contrast_id"] = "Undeclared post-hoc contrast"
        for records in (duplicate, unknown):
            for strict in (False, True):
                with self.subTest(strict=strict, last=records[-1]["contrast_id"]):
                    with self.assertRaises((ValueError, TypeError)):
                        ca.holm12(records, require_complete=strict)

    def test_nested_adjustment_or_rejection_fields_are_not_raw_records(self):
        for nested in ({"adjusted_p_value": {"numerator": 1, "denominator": 2}}, [{"reject": False}]):
            records = complete_records()[:3]
            records[0]["nested"] = nested
            with self.subTest(nested=nested):
                with self.assertRaises((ValueError, TypeError)):
                    ca.holm12(records, require_complete=False)


class ExecutionEndpointContract(unittest.TestCase):
    @staticmethod
    def fixture():
        families = ("fixed", "direct", "lagrangian", "shieldaware")
        selected = [f"{family}_{index:02d}" for family in families for index in (0, 1)]
        selection = {
            "schema": "PCI-DEVELOPMENT-FRONTIER-SELECTION-v1",
            "status": "DEVELOPMENT_ONLY_SELECTION",
            "study_domain": "warehouse",
            "primary_test_data_opened": False,
            "selected_candidate_ids": selected,
        }
        rows = [
            {"candidate_id": f"{family}_{index:02d}", "training_seed": seed,
             "family": family, "load": load, "replicate": replicate,
             "complete_drain": True, "n_agents": 4, "offered_requests": 100,
             "valid_unapplied_proofs": 10, "delivered_task_units": 80}
            for seed in range(62001, 62021)
            for family in families for index in (0, 1)
            for load in ("below_capacity", "overload") for replicate in (0, 1)
        ]
        return selection, rows

    def test_average_replicate_points_before_hypervolume(self):
        selection, rows = self.fixture()
        for row in rows:
            if row["family"] == "fixed":
                high = (row["candidate_id"] == "fixed_00") != (row["replicate"] == 0)
                row["valid_unapplied_proofs"] = 100 if high else 0
                row["delivered_task_units"] = 400 if high else 80
        result = ca.execution_endpoints(rows, selection)
        self.assertEqual(len(result), 20 * 2 * 4)
        for seed in range(62001, 62021):
            for load in ("below_capacity", "overload"):
                # Both policies' average point is(.5,.6), yielding area.3.
                # Averaging replicate-wise pair areas would instead yield.2.
                self.assertEqual(result[seed, load, "fixed"], F(3, 10))
                self.assertEqual(result[seed, load, "direct"], F(9, 50))

    def test_invalid_replicate_bounds_cannot_hide_inside_a_valid_average(self):
        for field, invalid in (("valid_unapplied_proofs", 200), ("delivered_task_units", 800)):
            selection, rows = self.fixture()
            rows[0][field] = invalid
            rows[1][field] = 0
            with self.subTest(field=field):
                with self.assertRaises((ValueError, TypeError)):
                    ca.execution_endpoints(rows, selection)

    def test_missing_or_duplicated_execution_cell_refused(self):
        selection, rows = self.fixture()
        for invalid in (rows[:-1], rows + [copy.deepcopy(rows[0])]):
            with self.assertRaises((ValueError, TypeError)):
                ca.execution_endpoints(invalid, selection)

    def test_common_paired_block_offer_exposure_enforced(self):
        selection, rows = self.fixture()
        rows[0]["offered_requests"] = 200
        with self.assertRaises((ValueError, TypeError)):
            ca.execution_endpoints(rows, selection)


class ProspectivelyClarifiedIntervals(unittest.TestCase):
    def test_fixed_synthetic_rng_and_interval_reference(self):
        # Synthetic differences1/20,...,20/20 have mean21/40 and variance7/80.
        # Independently evaluated approved PCG64/linear-percentile recipe gives
        # bounds2/5 and13/20; the seed is the first8 big-endian SHA256 bytes.
        values = [F(i, 20) for i in range(1, 21)]
        result = ca.paired_intervals(EXPECTED_IDS[0], values)
        bootstrap = result["bootstrap95"]
        self.assertEqual(bootstrap["seed"], 5241013097387328846)
        self.assertEqual(bootstrap["resamples"], 10000)
        self.assertEqual(bootstrap["quantile_method"], "linear")
        self.assertEqual(rational(bootstrap["lower"]), F(2, 5))
        self.assertEqual(rational(bootstrap["upper"]), F(13, 20))
        self.assertEqual(rational(result["paired_sample_variance"]), F(7, 80))
        self.assertEqual(result["paired_t95"]["df"], 19)
        self.assertAlmostEqual(result["paired_t95"]["lower"], 0.3865594715989873, places=12)
        self.assertAlmostEqual(result["paired_t95"]["upper"], 0.6634405284010128, places=12)
        self.assertAlmostEqual(result["dz"]["value"], 1.7748239349298849, places=12)
        self.assertIs(bootstrap["simultaneous"], False)
        self.assertIs(result["paired_t95"]["descriptive_sensitivity"], True)

    def test_zero_sample_variance_is_explicit_and_dz_undefined(self):
        result = ca.paired_intervals(EXPECTED_IDS[0], [F(1, 4)] * 20)
        self.assertEqual(rational(result["bootstrap95"]["lower"]), F(1, 4))
        self.assertEqual(rational(result["bootstrap95"]["upper"]), F(1, 4))
        self.assertEqual(result["paired_t95"]["status"], "DEGENERATE_ZERO_SAMPLE_VARIANCE")
        self.assertEqual(result["paired_t95"]["lower"], 0.25)
        self.assertEqual(result["paired_t95"]["upper"], 0.25)
        self.assertEqual(result["dz"]["status"], "UNDEFINED_ZERO_PAIRED_SD")
        self.assertIsNone(result["dz"]["value"])

    def test_canonical_seed_identity_and_order_are_enforced(self):
        for seeds in (list(range(62020, 62000, -1)), list(range(75001, 75021)), [62001] * 20):
            with self.subTest(seeds=seeds[:2]):
                with self.assertRaises((ValueError, TypeError)):
                    ca.contrast_statistics(EXPECTED_IDS[0], seeds, [F(0)] * 20)


class ScientificBoundaryControls(unittest.TestCase):
    def test_analysis_recipe_identity_is_bound(self):
        recipe = ca.verify_recipe()
        self.assertEqual(recipe["recipe_id"], "PCI-M3-CONFIRMATION-ANALYSIS-20261106-v1")
        self.assertEqual(recipe["canonical_contrast_ids"], list(EXPECTED_IDS))
        self.assertEqual(recipe["status"], "NEW_PROSPECTIVE_ANALYSIS_CLARIFICATION_NOT_PART_OF_PRIOR_PROTOCOL")

    def test_reformatted_recipe_copy_is_refused(self):
        with tempfile.TemporaryDirectory(prefix="pci_inference_synthetic_") as tmp:
            copied = Path(tmp) / "changed_recipe.json"
            copied.write_bytes((ROOT / "ANALYSIS_CLARIFICATION.json").read_bytes() + b"\n")
            with self.assertRaises((ValueError, TypeError)):
                ca.verify_recipe(copied)

    def test_unqualified_execution_or_combined_packet_refused_before_cohort_access(self):
        packet = {
            "schema": "PCI-CONFIRMATION-INFERENCE-RESULT-v1",
            "recipe_sha256": ca.RECIPE_SHA256,
            "source_map_sha256": ca.PINNED_SOURCE,
            "protocol_sha256": ca.PINNED_PROTOCOL,
            "contrasts": complete_records(),
        }
        with tempfile.TemporaryDirectory(prefix="pci_inference_synthetic_") as tmp:
            path = Path(tmp) / "fabricated_packet.json"
            for provenance in ({}, {"kind": "SOURCE_BOUND_COMPLETE_EXECUTION_COHORT"},
                               {"kind": "REVALIDATED_PARTIAL_LEARNING_PACKETS"}):
                with self.subTest(provenance=provenance):
                    packet["provenance"] = provenance
                    path.write_text(json.dumps(packet))
                    with patch.object(ca, "learning_packet", side_effect=AssertionError("unexpected cohort access")) as load:
                        with self.assertRaisesRegex(ValueError, "ADAPTER_NOT_QUALIFIED"):
                            ca.verified_learning_packet(path)
                        load.assert_not_called()


def main():
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = {
        "schema": "PCI-CONFIRMATION-INFERENCE-INDEPENDENT-CONTROLS-v1",
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "synthetic_only": True,
        "scientific_outcomes_accessed": False,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "test_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "core_source_sha256": hashlib.sha256(Path(ca.__file__).read_bytes()).hexdigest(),
    }
    (ROOT / "INDEPENDENT_TEST_RESULTS.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
