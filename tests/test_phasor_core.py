import unittest

import numpy as np

from reservoir import PhasorReservoir


class PhasorCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reservoir = PhasorReservoir(dimension=2048, seed=42)

    def test_codevectors_have_unit_magnitude(self) -> None:
        vector = self.reservoir.allocate_codevector("unit-magnitude-entity")
        self.assertEqual(vector.dtype, np.complex64)
        self.assertEqual(vector.nbytes, 16 * 1024)
        np.testing.assert_allclose(np.abs(vector), 1.0, atol=1e-6)

    def test_binding_is_invertible(self) -> None:
        entity = self.reservoir.allocate_codevector("invertible-entity")
        relation = self.reservoir.allocate_codevector("invertible-relation")
        recovered = self.reservoir.unbind(self.reservoir.bind(entity, relation), relation)
        self.assertGreater(self.reservoir.similarity(recovered, entity), 0.999)

    def test_independent_codevectors_are_quasi_orthogonal(self) -> None:
        first = self.reservoir.allocate_codevector("orthogonal-first")
        second = self.reservoir.allocate_codevector("orthogonal-second")
        self.assertLess(abs(self.reservoir.similarity(first, second)), 0.08)

    def test_binding_is_commutative(self) -> None:
        first = self.reservoir.allocate_codevector("commutative-first")
        second = self.reservoir.allocate_codevector("commutative-second")
        np.testing.assert_allclose(
            self.reservoir.bind(first, second),
            self.reservoir.bind(second, first),
            rtol=0.0,
            atol=1e-7,
        )

    def test_five_hop_path_unbinding_recovers_anchor(self) -> None:
        entities = [f"path-entity-{index}" for index in range(6)]
        relations = [f"path-relation-{index}" for index in range(5)]
        path = self.reservoir.bind_path(entities, relations)

        recovered = path
        for index in reversed(range(5)):
            recovered = self.reservoir.unbind(
                recovered,
                self.reservoir.allocate_codevector(f"entity:{entities[index + 1]}"),
            )
            recovered = self.reservoir.unbind(
                recovered,
                self.reservoir.allocate_codevector(f"relation:{relations[index]}"),
            )

        anchor = self.reservoir.allocate_codevector(f"entity:{entities[0]}")
        self.assertGreater(self.reservoir.similarity(recovered, anchor), 0.90)

    def test_twenty_triple_bundle_supports_individual_queries(self) -> None:
        triples = []
        targets = []
        query_keys = []
        for index in range(20):
            subject = self.reservoir.allocate_codevector(f"capacity-subject-{index}")
            relation = self.reservoir.allocate_codevector(f"capacity-relation-{index}")
            target = self.reservoir.allocate_codevector(f"capacity-target-{index}")
            triples.append(self.reservoir.bind(self.reservoir.bind(subject, relation), target))
            targets.append(target)
            query_keys.append(self.reservoir.bind(subject, relation))

        memory = self.reservoir.superpose(triples, normalize=False)
        retrieval_scores = []
        noise_scores = []
        for index, (query_key, target) in enumerate(zip(query_keys, targets)):
            retrieved = self.reservoir.unbind(memory, query_key)
            retrieval_scores.append(self.reservoir.similarity(retrieved, target))
            noise_scores.extend(
                self.reservoir.similarity(retrieved, other_target)
                for other_index, other_target in enumerate(targets)
                if other_index != index
            )

        noise_floor = float(np.mean(noise_scores))
        mean_retrieval = float(np.mean(retrieval_scores))
        # Keep the linear sum unprojected so its per-item retrieval signal is
        # preserved while measuring against unrelated target vectors.
        self.assertGreater(mean_retrieval - noise_floor, 0.35)
        self.assertTrue(all(score > noise_floor for score in retrieval_scores))


if __name__ == "__main__":
    unittest.main()
