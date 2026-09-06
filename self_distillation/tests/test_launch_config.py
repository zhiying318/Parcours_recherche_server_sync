import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import Trainer
from types import SimpleNamespace
from self_distillation.vlm_opsd.train import parse_args
from self_distillation.vlm_opsd.trainer import VLMOPSDTrainer


class LaunchConfigTest(unittest.TestCase):
    def test_epoch_and_step_precedence(self):
        self.assertEqual(parse_args([]).max_steps, 100)
        self.assertEqual(parse_args(['--num-train-epochs', '3']).max_steps, -1)
        self.assertEqual(parse_args(['--num-train-epochs', '3', '--max-steps', '20']).max_steps, 20)
        self.assertEqual(parse_args([]).report_to, 'wandb')
        self.assertNotEqual(parse_args([]).wandb_project, 'opsd-vlm')

    def test_container_batch_arithmetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, content in [('nvcc', '#!/bin/bash\nexit 0\n'),
                                  ('docker', '#!/bin/bash\nexit 0\n')]:
                path = root / 'bin' / name
                path.parent.mkdir(exist_ok=True)
                path.write_text(content)
                path.chmod(0o755)
            env = dict(os.environ, OPSD_CUDA_HOME=tmp, OPSD_VLM_HF_CACHE=tmp,
                       OPSD_VLM_PYTHON_PACKAGES=tmp, PATH=str(root/'bin')+':'+os.environ['PATH'])
            env.pop('OPSD_GLOBAL_BATCH_SIZE', None)
            script = 'self_distillation/vlm_opsd/run_docker.sh'
            for gpu, extra, target, expected in [
                ('0,1', ['--per-device-batch-size','2'], None, 'accumulation=1 global_batch=4'),
                ('0', ['--per-device-batch-size=2'], None, 'accumulation=1 global_batch=2'),
                ('0', ['--per-device-batch-size','2'], '4', 'accumulation=2 global_batch=4'),
                ('0', ['--per-device-batch-size','2','--gradient-accumulation-steps','2'], None, 'accumulation=2 global_batch=4'),
            ]:
                current = dict(env)
                if target:
                    current['OPSD_GLOBAL_BATCH_SIZE'] = target
                result = subprocess.run(['bash',script,gpu,*extra], env=current, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(expected, result.stdout)
            env['OPSD_GLOBAL_BATCH_SIZE'] = '2'
            result = subprocess.run(['bash',script,'0,1','--per-device-batch-size','2'], env=env, capture_output=True)
            self.assertEqual(result.returncode, 2)

    def test_metrics_average_accumulated_batches_and_clear(self):
        trainer = object.__new__(VLMOPSDTrainer)
        trainer._metric_batches = [{'raw_kl': 1.0}, {'raw_kl': 3.0}]
        trainer.accelerator = SimpleNamespace(device='cpu', reduce=lambda x, reduction: x)
        with patch.object(Trainer, 'log') as log, patch('torch.cuda.is_available', return_value=False):
            trainer.log({'loss': 2.0})
        self.assertEqual(log.call_args.args[0]['raw_kl'], 2.)
        self.assertEqual(trainer._metric_batches, [])
