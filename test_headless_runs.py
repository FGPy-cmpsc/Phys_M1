import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MAIN = ROOT / 'main.py'
OUTPUT_DIR = ROOT / 'test_outputs'


class HeadlessRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OUTPUT_DIR.mkdir(exist_ok=True)

    def _run_case(self, case_name, *args):
        env = os.environ.copy()
        env['MPLBACKEND'] = 'Agg'
        image_path = OUTPUT_DIR / f'{case_name}.png'
        proc = subprocess.run(
            [sys.executable, str(MAIN), '--headless', '--json', '--save-image', str(image_path), *args],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
        data = json.loads(proc.stdout)
        self.assertGreater(data['capacitance_f'], 0.0)
        self.assertGreater(data['n_total'], 0)
        self.assertLess(data['q1'] * data['q2'], 0.0)
        self.assertEqual(data['image_path'], str(image_path.resolve()))
        self.assertTrue(image_path.exists())
        self.assertGreater(image_path.stat().st_size, 0)
        return data

    def test_dense_air_gap_case(self):
        data = self._run_case(
            'dense_air_gap',
            '--l1', '0.10', '--w1', '0.10', '--t1', '0.003',
            '--l2', '0.10', '--w2', '0.10', '--t2', '0.003',
            '--gap', '0.008', '--voltage', '100', '--n', '13', '--eps', '1.0',
        )
        self.assertGreater(data['capacitance_pf'], 5.0)

    def test_fringing_case(self):
        data = self._run_case(
            'fringing',
            '--l1', '0.05', '--w1', '0.05', '--t1', '0.002',
            '--l2', '0.05', '--w2', '0.05', '--t2', '0.002',
            '--gap', '0.030', '--voltage', '60', '--n', '8', '--eps', '1.0',
        )
        self.assertGreater(data['capacitance_ratio'], 1.0)

    def test_asymmetric_plate_case(self):
        data = self._run_case(
            'asymmetric_plate',
            '--l1', '0.12', '--w1', '0.10', '--t1', '0.003',
            '--l2', '0.06', '--w2', '0.08', '--t2', '0.003',
            '--gap', '0.010', '--voltage', '80', '--n', '10', '--eps', '1.0',
        )
        self.assertGreater(data['capacitance_pf'], 1.0)

    def test_dielectric_case(self):
        data = self._run_case(
            'dielectric',
            '--l1', '0.10', '--w1', '0.10', '--t1', '0.003',
            '--l2', '0.10', '--w2', '0.10', '--t2', '0.003',
            '--gap', '0.008', '--voltage', '100', '--n', '13', '--eps', '4.5',
        )
        self.assertGreater(data['capacitance_pf'], 40.0)

    def test_low_voltage_insulation_case(self):
        data = self._run_case(
            'low_voltage_insulation',
            '--l1', '0.20', '--w1', '0.12', '--t1', '0.004',
            '--l2', '0.20', '--w2', '0.12', '--t2', '0.004',
            '--gap', '0.015', '--voltage', '24', '--n', '9', '--eps', '2.2',
        )
        self.assertGreater(data['capacitance_pf'], 10.0)


if __name__ == '__main__':
    unittest.main()
