import hashlib
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from download_models import download, verify
from resource_watch import reason
from validation import validate_job


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


if __name__ == '__main__': unittest.main()
