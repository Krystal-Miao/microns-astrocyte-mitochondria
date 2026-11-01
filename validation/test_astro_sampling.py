import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_astro_unet import batch
from train_astro_single_volume import replay_batch

class SamplingTest(unittest.TestCase):
    def test_trace_reconstructs_batch_without_changing_rng(self):
        rng=np.random.default_rng(7)
        a=rng.integers(0,256,(2,512,512),dtype=np.uint8)
        m=(a>200).astype(np.uint8)
        data={'sample_01':(a,m,[0,1])}
        records=[('sample_01',0)]
        trace=[]
        x,y=batch(data,records,np.random.default_rng(42),256,2,trace=trace)
        xx,yy=batch(data,records,np.random.default_rng(42),256,2)
        self.assertTrue(torch.equal(x,xx));self.assertTrue(torch.equal(y,yy))
        for i,t in enumerate(trace):
            self.assertEqual(t['z'],0)
            aa=a[t['z'],t['y']:t['y']+256,t['x']:t['x']+256]
            mm=m[t['z'],t['y']:t['y']+256,t['x']:t['x']+256]
            self.assertEqual(t['foreground_fraction'],float(mm.mean()))
            aa=np.rot90(aa,t['rotation_k'])
            if t['flip_horizontal']: aa=aa[:,::-1]
            np.testing.assert_allclose(x[i,0].numpy(),aa.astype(np.float32)/255)
        rx,ry,rt=replay_batch(data,trace)
        for i,t in enumerate(trace):
            sl=(t['z'],slice(t['y'],t['y']+256),slice(t['x'],t['x']+256))
            np.testing.assert_array_equal(rx[i,0].numpy(),a[sl].astype(np.float32)/255)
            np.testing.assert_array_equal(ry[i,0].numpy(),m[sl])
            self.assertEqual(rt[i]['rotation_k'],0)
            self.assertFalse(rt[i]['flip_horizontal'])
            for key in ['sample','z','x','y','size','foreground_fraction']:
                self.assertEqual(rt[i][key],t[key])

if __name__=='__main__': unittest.main()
