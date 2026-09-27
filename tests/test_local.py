import hashlib
import json
import pathlib
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from download_models import download, verify
from resource_watch import reason, run, process_usage
from validation import validate_job, validate_vace_keys, flatten_video_frames, validate_motion_controls
from assemble_preview import composite
from local_paths import validate_local_output
from segment_person import validate_prompts
from prepare_job import select_frames, validate_box, continuation_inputs, validate_mask_crop
from PIL import Image
import numpy as np
import psutil


class SourcePreparationTests(unittest.TestCase):
    def test_visible_character_cannot_be_silently_cropped(self):
        mask=Image.new('L',(100,100));mask.paste(255,(30,10,70,95))
        with self.assertRaisesRegex(ValueError,'outside the generation crop'):
            validate_mask_crop(mask,[20,0,80,80])
        validate_mask_crop(mask,[20,0,80,100],5)
        with self.assertRaises(ValueError):validate_mask_crop(mask,[30,10,70,95],1)

    def test_continuation_rejects_misalignment_and_reference_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            p=pathlib.Path(temp);(p/'generated').mkdir()
            ref=Image.new('RGB',(64,48),'white');ref.save(p/'reference.png')
            old=dict(engine='animate',width=64,height=48,crop=[0,0,64,48],frames=9,
                     source_sha256='a'*64,source_frame_indices=list(range(9)))
            (p/'config.json').write_text(json.dumps(old))
            for i in range(9):ref.save(p/'generated'/f'{i:03}.png')
            current={**old,'source_frame_indices':list(range(4,13))}
            selected=continuation_inputs(p,current,ref,5)
            self.assertEqual([x.name for x in selected],['004.png','005.png','006.png','007.png','008.png'])
            with self.assertRaisesRegex(ValueError,'align'):
                continuation_inputs(p,{**current,'source_frame_indices':list(range(5,14))},ref,5)
            with self.assertRaisesRegex(ValueError,'reference image changed'):
                continuation_inputs(p,current,Image.new('RGB',(64,48),'black'),5)
            (p/'generated/008.png').unlink()
            with self.assertRaisesRegex(ValueError,'generation is incomplete'):
                continuation_inputs(p,current,ref,5)

    def test_timestamps_and_bounds(self):
        times = [i/25 for i in range(25)]
        self.assertEqual(select_frames(times, .16, 5, 12.5, 1), [4,6,8,10,12])
        self.assertEqual(select_frames(times,.88,5,25,1,pad_last=True),[22,23,24,24,24])
        with self.assertRaisesRegex(ValueError,'three final padding'):
            select_frames(times,.88,9,25,1,pad_last=True)
        with self.assertRaisesRegex(ValueError,'duplicates'):
            select_frames(times,0,5,50,1,pad_last=True)
        for args in [(times,.8,5,12.5,1), (times,0,5,50,1),
                     ([0,.08,.04],0,2,25,1), (times,0,5,25,float('nan'))]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                select_frames(*args)
        with self.assertRaises(ValueError):validate_box([0,0,65,48],(64,48))

    def test_real_decode_prepare_composite_and_changed_source(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            p=pathlib.Path(temp); source=p/'source.mp4'; masks=p/'masks';masks.mkdir()
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i',
                'testsrc2=size=64x48:rate=25:duration=1','-c:v','libx264',str(source)],check=True)
            Image.new('RGB',(64,48),'white').save(p/'reference.png')
            for i in range(25):Image.new('L',(64,48),255).save(masks/f'{i:05}.png')
            cfg=dict(width=64,height=48,frames=5,fps=12.5,start=.16,crop=[0,0,64,48],
                     steps=20,cfg=1,positive='test person',negative='',seed=1)
            (p/'settings.json').write_text(json.dumps(cfg));job=p/'job'
            subprocess.run([sys.executable,str(root/'scripts/prepare_job.py'),
                '--source',str(source),'--reference',str(p/'reference.png'),
                '--masks',str(masks),'--settings',str(p/'settings.json'),
                '--output',str(job)],check=True,capture_output=True)
            prepared=json.loads((job/'config.json').read_text())
            self.assertEqual(prepared['source_frame_indices'],[4,6,8,10,12])
            self.assertFalse((job/'INCOMPLETE').exists())
            (job/'generated').mkdir()
            for frame in (job/'frames').glob('*.png'):
                (job/'generated'/frame.name).write_bytes(frame.read_bytes())
            command=[sys.executable,str(root/'scripts/assemble_preview.py'),str(job),str(source)]
            subprocess.run(command,check=True,capture_output=True)
            for i in range(5):
                self.assertTrue(np.array_equal(np.asarray(Image.open(job/'source-frames'/f'{i:03}.png')),
                                               np.asarray(Image.open(job/'composite'/f'{i:03}.png'))))
            tail=p/'tail';(p/'settings.json').write_text(json.dumps({**cfg,'start':.88,'fps':25,'pad_last':True}))
            subprocess.run([sys.executable,str(root/'scripts/prepare_job.py'),
                '--source',str(source),'--reference',str(p/'reference.png'),
                '--masks',str(masks),'--settings',str(p/'settings.json'),
                '--output',str(tail)],check=True,capture_output=True)
            tail_cfg=json.loads((tail/'config.json').read_text())
            self.assertEqual(tail_cfg['padded_tail_frames'],2)
            self.assertAlmostEqual(tail_cfg['output_duration'],.12)
            (tail/'generated').mkdir()
            for frame in (tail/'frames').glob('*.png'):
                (tail/'generated'/frame.name).write_bytes(frame.read_bytes())
            subprocess.run([sys.executable,str(root/'scripts/assemble_preview.py'),str(tail),str(source)],
                           check=True,capture_output=True)
            metadata=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries',
                'stream=nb_frames,duration','-of','json',str(tail/'replacement-preview.mp4')]))
            self.assertEqual(metadata['streams'][0]['nb_frames'],'3')
            self.assertAlmostEqual(float(metadata['streams'][0]['duration']),.12)
            with source.open('ab') as stream:stream.write(b'changed')
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Source video changed',result.stderr)
            (job/'INCOMPLETE').write_text('interrupted')
            with self.assertRaisesRegex(ValueError,'Incomplete job preparation'):
                validate_job(job,prepared)


class SegmentationPromptTests(unittest.TestCase):
    def test_reviewed_positive_and_negative_points(self):
        validate_prompts([{'frame': 0, 'points': [[20, 30], [50, 50]],
                           'labels': [1, 0]}], 13, (100, 80))

    def test_invalid_coordinates_and_frame_indices(self):
        for frame, point in [(13, [20, 30]), (-1, [20, 30]),
                             (0, [100, 20]), (0, [float('nan'), 20])]:
            with self.subTest(frame=frame, point=point), self.assertRaises(ValueError):
                validate_prompts([{'frame': frame, 'points': [point], 'labels': [1]}],
                                 13, (100, 80))

    def test_missing_initial_or_positive_prompt_rejected(self):
        for prompts in [[], [{'frame': 1, 'points': [[20, 30]], 'labels': [1]}],
                        [{'frame': 0, 'points': [[20, 30]], 'labels': [0]}],
                        [{'frame': 0, 'points': [[20, 30]], 'labels': []}]]:
            with self.subTest(prompts=prompts), self.assertRaises(ValueError):
                validate_prompts(prompts, 13, (100, 80))


class LocalOutputTests(unittest.TestCase):
    def test_nonsynced_path_allowed(self):
        with tempfile.TemporaryDirectory() as temp:
            validate_local_output(pathlib.Path(temp)/'job')

    @unittest.skipUnless(sys.platform == 'darwin', 'macOS extended attributes')
    def test_dropbox_requires_inherited_exclusion_and_resolves_symlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            base=pathlib.Path(temp); cloud=base/'Dropbox'; cloud.mkdir()
            alias=base/'alias'; alias.symlink_to(cloud, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'without an ignore'):
                validate_local_output(alias/'job')
            subprocess.run(['/usr/bin/xattr','-w','com.dropbox.ignored','1',str(cloud)],check=True)
            validate_local_output(alias/'job')
            subprocess.run(['/usr/bin/xattr','-w','com.dropbox.ignored','0',str(cloud)],check=True)
            with self.assertRaises(ValueError):validate_local_output(alias/'job')


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
    def test_cli_sigterm_reaps_inference_process_group(self):
        scripts=pathlib.Path(__file__).resolve().parents[1]/'scripts'
        with tempfile.TemporaryDirectory() as temp:
            p=pathlib.Path(temp); child_pid=p/'child'; grandchild_pid=p/'grandchild'
            grandchild=f'import os,time,pathlib;pathlib.Path({str(grandchild_pid)!r}).write_text(str(os.getpid()));time.sleep(60)'
            child=f'import os,time,pathlib,subprocess,sys;pathlib.Path({str(child_pid)!r}).write_text(str(os.getpid()));subprocess.Popen([sys.executable,"-c",{grandchild!r}]);time.sleep(60)'
            launcher='import resource_watch as r,sys,time;r.sample=lambda:dict(time=time.time(),pressure=1,swap=0,available_disk=100_000_000_000);sys.exit(r.main())'
            supervisor=subprocess.Popen([sys.executable,'-c',launcher,str(p/'resources'),
                sys.executable,'-c',child],env={**os.environ,'PYTHONPATH':str(scripts)},
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            descendants=[]
            try:
                deadline=time.monotonic()+5
                while not grandchild_pid.exists() and time.monotonic()<deadline:
                    time.sleep(.02)
                self.assertTrue(grandchild_pid.exists(),'Test child did not start')
                descendants=[int(child_pid.read_text()),int(grandchild_pid.read_text())]
                supervisor.terminate()
                _,error=supervisor.communicate(timeout=5)
                self.assertEqual(supervisor.returncode,130,error)
                self.assertIn('process group terminated',error)
                for pid in descendants:
                    try:
                        process=psutil.Process(pid)
                        self.assertEqual(process.status(),psutil.STATUS_ZOMBIE)
                    except psutil.NoSuchProcess:
                        pass
            finally:
                if supervisor.poll() is None:
                    supervisor.kill();supervisor.communicate()
                for pid in descendants:
                    try:psutil.Process(pid).kill()
                    except psutil.NoSuchProcess:pass

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

    def test_invalid_text_precision_fails_before_input_access(self):
        with self.assertRaisesRegex(ValueError, "precision"):
            validate_job("missing", {"text_dtype": "fp8"})

    def test_unknown_or_wrong_architecture_model_rejected(self):
        for config in ({'model': 'unverified.gguf'}, {'engine': 'unknown'},
                       {'engine': 'vace', 'model': 'Wan2.2-Animate-14B-Q2_K.gguf'}):
            with self.assertRaisesRegex(ValueError, 'model or engine'):
                validate_job('missing', config)

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

    def test_missing_continuation_refused_before_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            root=pathlib.Path(temp);cfg=dict(frames=9,width=64,height=48,continue_motion_frames=5)
            for folder,size in [('poses',(64,48)),('faces',(512,512))]:
                (root/folder).mkdir()
                for i in range(9):Image.new('RGB',size).save(root/folder/f'{i:03}.png')
            with self.assertRaisesRegex(ValueError,'continuation'):
                validate_motion_controls(root,cfg)
            (root/'continuation').mkdir()
            for i in range(5):Image.new('RGB',(64,48)).save(root/'continuation'/f'{i:03}.png')
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

    def test_resized_generation_preserves_source_outside_crop(self):
        source = Image.new('RGB', (12,12), (20,40,60))
        generated = Image.new('RGB', (8,8), (200,100,50))
        mask = Image.new('L', (8,8), 255)
        result = np.asarray(composite(source, generated, mask, (4,4,8,8), True))
        expected = np.asarray(source).copy()
        expected[4:8,4:8] = (200,100,50)
        np.testing.assert_array_equal(result, expected)
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            composite(source, generated, Image.new('L',(4,4)), (4,4,8,8), True)

    def test_invalid_crop_rejected(self):
        with self.assertRaises(ValueError):
            composite(Image.new('RGB',(4,4)),Image.new('RGB',(4,4)),Image.new('L',(4,4)),(2,2,6,6))


if __name__ == '__main__': unittest.main()
