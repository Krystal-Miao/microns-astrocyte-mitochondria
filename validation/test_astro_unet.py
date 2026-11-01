"""Run with: python -m unittest discover -s validation -p 'test_astro_unet.py'."""
import sys
import unittest
from unittest.mock import patch
import tempfile
import json
from pathlib import Path
import numpy as np
import torch
import tifffile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_astro_unet import UNet, loss_fn, metrics, make_split, batch, evaluate, main

class BaselineTests(unittest.TestCase):
    def test_metrics_known_overlap(self):
        score=metrics(3,1,2)
        self.assertAlmostEqual(score['dice'],2/3)
        self.assertAlmostEqual(score['iou'],0.5)
        self.assertEqual(metrics(0,0,0)['dice'],1)
        self.assertEqual(metrics(0,0,4)['dice'],0)

    def test_shape_and_backward(self):
        torch.set_num_threads(2)
        model=UNet(2)
        x=torch.rand(2,1,33,41)
        y=torch.randint(0,2,x.shape).float()
        output=model(x)
        self.assertEqual(output.shape,x.shape)
        loss=loss_fn(output,y);loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_volume_split_and_guard_band(self):
        data={f'sample_{i:02d}':(None,None,[z for z in range(128) if z!=9]) for i in range(1,4)}
        for fold in range(1,4):
            train,val,test=make_split(data,fold)
            self.assertTrue(all(n==f'sample_{fold:02d}' for n,z in test))
            self.assertFalse({n for n,z in train+val}&{n for n,z in test})
            self.assertTrue(all(z<96 for n,z in train))
            self.assertTrue(all(z>=112 for n,z in val))
            self.assertFalse(set(train)&set(val))
            self.assertTrue(all(z!=9 for n,z in train+val+test))

    def test_paired_augmentation(self):
        rng=np.random.default_rng(5)
        mask=rng.integers(0,2,(1,512,512),dtype=np.uint8)
        data={'a':(mask*255,mask,[0])}
        x,y=batch(data,[('a',0)],rng,64,8)
        self.assertTrue(torch.equal(x,y))
        self.assertEqual(x.shape,(8,1,64,64))

    def test_evaluation_exports_original_z_mapping(self):
        class Threshold(torch.nn.Module):
            def forward(self,x): return (x-0.5)*20
        mask=np.zeros((3,16,16),dtype=np.uint8)
        mask[0,:4]=1;mask[2,:8]=1
        data={'sample':(mask*255,mask,[0,2])}
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            score=evaluate(Threshold(),data,[('sample',0),('sample',2)],'cpu',out)
            self.assertEqual(score['dice'],1)
            self.assertEqual(score['tp'],192)
            self.assertAlmostEqual(score['predicted_foreground_fraction'],192/512)
            self.assertAlmostEqual(score['labeled_foreground_fraction'],192/512)
            self.assertTrue(np.isfinite(score['loss']))
            saved=tifffile.imread(out/'sample_prediction_valid_slices.tif')
            np.testing.assert_array_equal(saved,mask[[0,2]])
            mapping=json.loads((out/'sample_prediction_z.json').read_text())
            self.assertEqual(mapping['source_z_per_output_plane'],[0,2])

    def test_final_checkpoint_used_when_best_is_earlier(self):
        a=np.zeros((128,512,512),dtype=np.uint8)
        data={f'sample_{i:02d}':(a,a,list(range(128))) for i in range(1,4)}
        def fake_evaluate(model,data,records,device,out=None):
            # First validation is best, final validation is worse.
            score=metrics(1,0,0) if fake_evaluate.calls==0 else metrics(0,0,1)
            fake_evaluate.calls+=1
            return {**score,'loss':1.0,'predicted_foreground_fraction':0.0,
                    'labeled_foreground_fraction':0.1}
        fake_evaluate.calls=0
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'run'
            argv=['train','--out',str(out),'--fold','3','--steps','101',
                  '--width','1','--patch','8','--batch-size','1','--device','cpu']
            with patch.object(sys,'argv',argv), patch('train_astro_unet.load_data',return_value=(data,{})), patch('train_astro_unet.evaluate',side_effect=fake_evaluate), patch('builtins.print'):
                main()
            best=torch.load(out/'fold_3/best.pt',weights_only=True)
            last=torch.load(out/'fold_3/last.pt',weights_only=True)
            result=json.loads((out/'fold_3/result.json').read_text())
            self.assertEqual(best['step'],100)
            self.assertEqual(last['step'],101)
            self.assertEqual(result['selected_step'],101)
            self.assertEqual(result['test_checkpoint'],'last.pt')
            self.assertEqual(result['validation_dice'],0.0)
            history=json.loads((out/'fold_3/history.json').read_text())
            self.assertIn('training_monitor',history[-1])
            self.assertIn('loss',history[-1]['validation'])

if __name__=='__main__': unittest.main()
