import hashlib
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from download_models import download, verify
from resource_watch import reason, run, process_usage
from validation import validate_job, validate_vace_keys, flatten_video_frames, validate_motion_controls
from assemble_preview import composite
from PIL import Image
import numpy as np


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.item = dict(destination='weights/x.gguf', size=4,
                         sha256=hashlib.sha256(b'test').hexdigest(),
                         repo='example/model', revision='a'*40, filename='x.gguf')

    def test_missing_corrupt_verified(self):
        p = self.root/'x'; self.assertFalse(verify(p,self.item))
        p.write_bytes(b'fail'); self.assertFalse(verify(p,self.item))
        p.write_bytes(b'test'); self.assertTrue(verify(p,self.item))

    def test_interrupted_download_preserves_partial_for_resume(self):
        dest = self.root/self.item['destination'];dest.parent.mkdir()
        part = dest.with_suffix('.gguf.part');part.write_bytes(b'te')
        with patch('download_models.subprocess.run', side_effect=subprocess.CalledProcessError(18,'curl')) as call:
            with self.assertRaises(subprocess.CalledProcessError): download(self.item,self.root)
        self.assertEqual(part.read_bytes(),b'te');self.assertFalse(dest.exists())
        self.assertIn('--continue-at', call.call_args.args[0])
        with patch('download_models.subprocess.run', side_effect=lambda *a,**k: part.write_bytes(b'test')):
            download(self.item,self.root)
        self.assertTrue(verify(dest,self.item));self.assertFalse(part.exists())

    def test_corrupt_existing_never_overwritten(self):
        dest=self.root/self.item['destination'];dest.parent.mkdir();dest.write_bytes(b'fail')
        with patch('download_models.subprocess.run') as call:
            with self.assertRaises(RuntimeError): download(self.item,self.root)
        call.assert_not_called();self.assertEqual(dest.read_bytes(),b'fail')


class ResourceTests(unittest.TestCase):
    def test_denied_process_inventory_does_not_stop_supervision(self):
        from unittest.mock import Mock
        process = Mock()
        process.children.side_effect = PermissionError('restricted inventory')
        process.memory_info.return_value.rss = 123
        process.cpu_times.return_value.user = 2
        process.cpu_times.return_value.system = 1
        with patch('resource_watch.psutil.Process', return_value=process):
            usage = process_usage(12345)
        self.assertEqual(usage['usage_scope'], 'direct_child_only')
        self.assertEqual(usage['rss'], 123)
        self.assertEqual(usage['cpu_seconds'], 3)

    def test_motion_controls_missing_and_wrong_dimensions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            cfg = dict(frames=1,width=192,height=256)
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                validate_motion_controls(root,cfg)
            for folder,size in [('poses',(192,256)),('faces',(128,128))]:
                (root/folder).mkdir()
                Image.new('RGB',size).save(root/folder/'000.png')
            with self.assertRaisesRegex(ValueError, 'dimensions'):
                validate_motion_controls(root,cfg)
            Image.new('RGB',(512,512)).save(root/'faces/000.png')
            validate_motion_controls(root,cfg)

    def record(self, pressure=1, swap=0, disk=100_000_000_000):
        return dict(pressure=pressure,swap=swap,available_disk=disk)

    def test_warning_is_not_critical(self):
        self.assertIsNone(reason([self.record(2)]*7))

    def test_sustained_critical_pressure(self):
        self.assertIsNone(reason([self.record(4)]*2))
        self.assertIn('critical',reason([self.record(4)]*3))

    def test_disk_and_swap_failures(self):
        self.assertIn('reserve',reason([self.record(disk=29_000_000_000)]))
        self.assertIn('Swap',reason([self.record()]*6+[self.record(swap=3_000_000_000)]))

    def test_invalid_dimensions_fail_before_file_reads(self):
        with self.assertRaises(ValueError):validate_job('/unused',dict(width=191,height=256))

    def test_incomplete_vace_cannot_silently_become_text_to_video(self):
        with self.assertRaisesRegex(ValueError,'vace_patch_embedding.weight'):
            validate_vace_keys(['vace_patch_embedding.bias','vace_blocks.0.before_proj.weight','vace_blocks.0.after_proj.weight'])
        validate_vace_keys(['vace_patch_embedding.weight','vace_patch_embedding.bias','vace_blocks.0.before_proj.weight','vace_blocks.0.after_proj.weight'])

    def test_resource_stop_terminates_child_and_allows_recovery(self):
        real_popen = subprocess.Popen
        children = []
        def launch(*args, **kwargs):
            p=real_popen(*args, **kwargs);children.append(p);return p
        with tempfile.TemporaryDirectory() as temp:
            log=pathlib.Path(temp)/'resources.jsonl'
            with patch('resource_watch.sample', return_value=self.record(4)), patch('resource_watch.time.sleep'), patch('resource_watch.subprocess.Popen',side_effect=launch):
                with self.assertRaisesRegex(RuntimeError,'critical'):
                    run([sys.executable,'-c','import time;time.sleep(30)'],log)
            self.assertIsNotNone(children[0].poll())
            with patch('resource_watch.sample', return_value=self.record()), patch('resource_watch.time.sleep'):
                self.assertEqual(run([sys.executable,'-c','raise SystemExit(7)'],log),7)
                self.assertEqual(run([sys.executable,'-c','pass'],log),0)

    def test_keyboard_cancellation_terminates_child(self):
        real_popen = subprocess.Popen
        children=[]
        def launch(*args,**kwargs):
            p=real_popen(*args,**kwargs);children.append(p);return p
        with tempfile.TemporaryDirectory() as temp, patch('resource_watch.sample',return_value=self.record()), patch('resource_watch.subprocess.Popen',side_effect=launch), patch('resource_watch.time.sleep',side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                run([sys.executable,'-c','import time;time.sleep(30)'],pathlib.Path(temp)/'log')
        self.assertIsNotNone(children[0].poll())


class CompositeTests(unittest.TestCase):
    def test_native_video_batch_is_flattened_without_reordering(self):
        a=np.arange(2*5*16*16*3).reshape(2,5,16,16,3)
        result=flatten_video_frames(a,(10,16,16,3))
        self.assertTrue(np.array_equal(result[5],a[1,0]))
        with self.assertRaises(ValueError):flatten_video_frames(a,(5,16,16,3))

    def test_untouched_pixels_are_identical(self):
        source=Image.fromarray(np.random.default_rng(1).integers(0,256,(12,16,3),dtype=np.uint8))
        generated=Image.new('RGB',(4,4),'red');mask=Image.new('L',(4,4),0)
        mask.putpixel((1,1),255)
        result=np.asarray(composite(source,generated,mask,(4,4,8,8)))
        changed=np.any(result!=np.asarray(source),axis=2)
        self.assertEqual(np.argwhere(changed).tolist(),[[5,5]])

    def test_invalid_crop_rejected(self):
        with self.assertRaises(ValueError):
            composite(Image.new('RGB',(4,4)),Image.new('RGB',(4,4)),Image.new('L',(4,4)),(2,2,6,6))


if __name__ == '__main__': unittest.main()
